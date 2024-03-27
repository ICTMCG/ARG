import os
import torch
import tqdm
import time
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader
from utils.utils import get_monthly_path, get_tensorboard_writer, process_test_results

class ARGModel(torch.nn.Module):
    def __init__(self, config):
        super(ARGModel, self).__init__()

        self.bert_content = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)

        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_FTR.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])

        self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.score_mapper_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.score_mapper_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.simple_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))
        self.simple_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))

        self.content_attention = MaskAttention(config['emb_dim'])    

        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

    
    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']

        FTR_2, FTR_2_masks = kwargs['FTR_2'], kwargs['FTR_2_masks']
        FTR_3, FTR_3_masks = kwargs['FTR_3'], kwargs['FTR_3_masks']

        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        content_feature_1, content_feature_2 = content_feature, content_feature

        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask = FTR_2_masks)[0]
        FTR_3_feature = self.bert_FTR(FTR_3, attention_mask = FTR_3_masks)[0]

        mutual_content_FTR_2, _ = self.cross_attention_content_2( \
            content_feature_2, FTR_2_feature, content_masks)
        expert_2 = torch.mean(mutual_content_FTR_2, dim=1)
    
        mutual_content_FTR_3, _ = self.cross_attention_content_3( \
            content_feature_2, FTR_3_feature, content_masks)
        expert_3 = torch.mean(mutual_content_FTR_3, dim=1)

        mutual_FTR_content_2, _ = self.cross_attention_ftr_2( \
            FTR_2_feature, content_feature_2, FTR_2_masks)
        mutual_FTR_content_2 = torch.mean(mutual_FTR_content_2, dim=1)

        mutual_FTR_content_3, _ = self.cross_attention_ftr_3( \
            FTR_3_feature, content_feature_2, FTR_3_masks)
        mutual_FTR_content_3 = torch.mean(mutual_FTR_content_3, dim=1)

        hard_ftr_2_pred = self.hard_mlp_ftr_2(mutual_FTR_content_2).squeeze(1)
        hard_ftr_3_pred = self.hard_mlp_ftr_3(mutual_FTR_content_3).squeeze(1)

        simple_ftr_2_pred = self.simple_mlp_ftr_2(self.simple_ftr_2_attention(FTR_2_feature)[0]).squeeze(1)
        simple_ftr_3_pred = self.simple_mlp_ftr_3(self.simple_ftr_3_attention(FTR_3_feature)[0]).squeeze(1)    

        attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)

        reweight_score_ftr_2 = self.score_mapper_ftr_2(mutual_FTR_content_2)
        reweight_score_ftr_3 = self.score_mapper_ftr_3(mutual_FTR_content_3)

        reweight_expert_2 = reweight_score_ftr_2 * expert_2
        reweight_expert_3 = reweight_score_ftr_3 * expert_3

        all_feature = torch.cat(
            (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1)), 
            dim = 1
        )
        final_feature, _ = self.aggregator(all_feature)

        label_pred = self.mlp(final_feature)
        gate_value = torch.concat([
            reweight_score_ftr_2,
            reweight_score_ftr_3
        ], dim=1)

        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content,
            'ftr_2_feature': reweight_expert_2,
            'ftr_3_feature': reweight_expert_3            
        }

        res['hard_ftr_2_pred'] = hard_ftr_2_pred
        res['hard_ftr_3_pred'] = hard_ftr_3_pred

        res['simple_ftr_2_pred'] = simple_ftr_2_pred
        res['simple_ftr_3_pred'] = simple_ftr_3_pred

        return res


class Trainer():
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer
        self.num_expert = 2
        
        self.save_path = os.path.join(
            self.config['save_param_dir'],
            self.config['model_name']+'_'+self.config['data_name'],
            str(self.config['month']))
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        

    def train(self, logger = None):
        st_tm = time.time()
        writer = self.writer

        if(logger):
            logger.info('start training......')
        print('\n\n')
        print('==================== start training ====================')

        self.model = ARGModel(self.config)

        if self.config['use_cuda']:
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])

        train_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'train.json')
        train_loader = get_dataloader(
            train_path, 
            self.config['max_len'], 
            self.config['batchsize'], 
            shuffle=True, 
            bert_path=self.config['bert_path'], 
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        val_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'val.json')
        val_loader = get_dataloader(
            val_path, 
            self.config['max_len'], 
            self.config['batchsize'], 
            shuffle=False, 
            bert_path=self.config['bert_path'], 
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        test_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'test.json')
        test_future_loader = get_dataloader(
            test_path, 
            self.config['max_len'], 
            self.config['batchsize'], 
            shuffle=False, 
            bert_path=self.config['bert_path'],
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        ed_tm = time.time()
        print('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
        for epoch in range(self.config['epoch']):
            print('---------- epoch {} ----------'.format(epoch))
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss_classify = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                label = batch_data['label']

                hard_ftr_2_label = batch_data['FTR_2_acc']
                hard_ftr_3_label = batch_data['FTR_3_acc']

                simple_ftr_2_label = batch_data['FTR_2_pred']
                simple_ftr_3_label = batch_data['FTR_3_pred']
                
                batch_input_data = {**self.config, **batch_data}

                res = self.model(**batch_input_data)
                loss_classify = loss_fn(res['classify_pred'], label.float())

                loss_hard_aux_fn = torch.nn.BCELoss()
                loss_hard_aux = loss_hard_aux_fn(res['hard_ftr_2_pred'], hard_ftr_2_label.float()) + loss_hard_aux_fn(res['hard_ftr_3_pred'], hard_ftr_3_label.float())
            
                loss_simple_aux_fn = torch.nn.CrossEntropyLoss()
                loss_simple_aux = loss_simple_aux_fn(res['simple_ftr_2_pred'], simple_ftr_2_label.long()) + loss_simple_aux_fn(res['simple_ftr_3_pred'], simple_ftr_3_label.long())

                loss = loss_classify
                loss += self.config['model']['rationale_usefulness_evaluator_weight'] * loss_hard_aux / self.num_expert
                loss += self.config['model']['llm_judgment_predictor_weight'] * loss_simple_aux / self.num_expert

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss_classify.add(loss_classify.item())

            print('----- in val progress... -----')
            results, val_aux_info = self.test(val_loader)
            mark = recorder.add(results)
            print()

            # tensorlog
            writer.add_scalar('month_'+str(self.config['month'])+'/train_loss', avg_loss_classify.item(), global_step=epoch)
            writer.add_scalars('month_'+str(self.config['month'])+'/test', results, global_step=epoch)

            # logger
            if(logger):
                logger.info('---------- epoch {} ----------'.format(epoch))
                logger.info('train loss classify: {}'.format(avg_loss_classify.item()))
                logger.info('\n')

                logger.info('val loss classify: {}'.format(val_aux_info['val_avg_loss_classify'].item()))

                logger.info('val result: {}'.format(results))
                logger.info('\n')

            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_bert.pkl'))
            if mark == 'esc':
                break
            else:
                continue

        test_dir = os.path.join(
            './logs/test/',
            self.config['model_name'] + '_' + self.config['data_name']
        )
        os.makedirs(test_dir, exist_ok=True)
        test_res_path = os.path.join(
            test_dir,
            'month_' + str(self.config['month']) + '.json'
        )

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bert.pkl')))
        future_results, label, pred, id, ae, acc = self.predict(test_future_loader)

        writer.add_scalars('month_'+str(self.config['month'])+'/test', future_results)
        
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, avg test score: {}.\n\n".format(self.config['lr'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bert.pkl'), epoch


    def test(self, dataloader):
        loss_fn = torch.nn.BCELoss()
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        avg_loss_classify = Averager()
        
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                batch_label = batch_data['label']

                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)

                loss_classify = loss_fn(res['classify_pred'], batch_label.float())
            
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(res['classify_pred'].detach().cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item())

        aux_info = {
            'val_avg_loss_classify': avg_loss_classify
        }

        return metrics(label, pred), aux_info


    def predict(self, dataloader):
        if self.config['eval_mode']:
            self.model = ARGModel(self.config)
            if self.config['use_cuda']:
                self.model = self.model.cuda()
            print('========== in test process ==========')
            print('now load in test model...')
            self.model.load_state_dict(torch.load(self.config['eval_model_path']))
        pred = []
        label = []
        id = []
        ae = []
        accuracy = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                batch_pred = res['classify_pred']

                cur_labels = batch_label.detach().cpu().numpy().tolist()
                cur_preds = batch_pred.detach().cpu().numpy().tolist()
                label.extend(cur_labels)
                pred.extend(cur_preds)
                ae_list = []
                for index in range(len(cur_labels)):
                    ae_list.append(abs(cur_preds[index] - cur_labels[index]))
                accuracy_list = [1 if ae<0.5 else 0 for ae in ae_list]
                ae.extend(ae_list)
                accuracy.extend(accuracy_list)
        
        return metrics(label, pred), label, pred, id, ae, accuracy
