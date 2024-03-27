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
from .arg import ARGModel

class ARGDModel(torch.nn.Module):
    def __init__(self, config):
        super(ARGDModel, self).__init__()

        self.teacher = ARGModel(config)

        self.bert = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.transformer = Block(dim=config['emb_dim'], num_heads=4)
        self.attention = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])

        self.bert.load_state_dict(self.teacher.bert_content.state_dict())
        self.mlp.load_state_dict(self.teacher.mlp.state_dict())

        self.params = list(self.transformer.parameters()) + list(self.attention.parameters()) + list(self.mlp.parameters())

        print("----- model initiating finish -----")
    
    def forward(self, **kwargs):
        teacher_res = self.teacher(**kwargs)
        t_final_feature, t_content_feature =  teacher_res['final_feature'], teacher_res['content_feature']

        content, content_masks = kwargs['content'], kwargs['content_masks']
        content_feature = self.bert(content, attention_mask = content_masks)[0]

        final_content = self.transformer(content_feature)
        final_feature, _ = self.attention(final_content)
    
        label_pred = self.mlp(final_feature)

        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            's_final_feature': final_feature,
            't_final_feature': t_final_feature,
            't_content_feature': t_content_feature,  
        }

        return res


class Trainer():
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer
        
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

        self.model = ARGDModel(self.config)
        self.model.teacher.load_state_dict(torch.load(self.config['teacher_path']))
        if self.config['use_cuda']:
            self.model = self.model.cuda()

        loss_crit = nn.MSELoss()
        loss_fn = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(params = self.model.params, lr = self.config['lr'], weight_decay = self.config['weight_decay'])
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

            train_avg_loss = Averager()
            train_avg_loss_classify = Averager()
            train_avg_loss_final = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                label = batch_data['label']
                
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)

                # loss cal
                loss_classify = loss_fn(res['classify_pred'], label.float())
                loss_final = loss_crit(res['s_final_feature'], res['t_final_feature'])
            
                # loss reweight
                loss = loss_classify + self.config['model']['kd_loss_weight'] * loss_final

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record loss
                train_avg_loss.add(loss.item())
                train_avg_loss_classify.add(loss_classify.item())
                train_avg_loss_final.add(torch.mean(loss_final).item())
                

            print('----- in val progress... -----')
            results, val_aux_info = self.test(val_loader)
            mark = recorder.add(results)
            print()

            writer.add_scalar('month_'+str(self.config['month'])+'/train_loss', train_avg_loss_classify.item(), global_step=epoch)
            writer.add_scalars('month_'+str(self.config['month'])+'/test', results, global_step=epoch)

            if(logger):
                logger.info('---------- epoch {} ----------'.format(epoch))
                logger.info('train loss classify: {}'.format(train_avg_loss_classify.item()))
                logger.info('train loss final: {}'.format(train_avg_loss_final.item()))
                logger.info('\n')
                logger.info('val loss classify: {}'.format(val_aux_info['val_avg_loss_classify'].item()))
                logger.info('val loss final: {}'.format(val_aux_info['val_avg_loss_final'].item()))

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
        loss_crit = nn.MSELoss(reduction = 'none')

        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        val_avg_loss_classify = Averager()
        val_avg_loss_final = Averager()
        
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

                loss_final = loss_crit(res['s_final_feature'], res['t_final_feature'])

                val_avg_loss_classify.add(loss_classify.item())
                val_avg_loss_final.add(torch.mean(loss_final).item())

        aux_info = {
            'val_avg_loss_classify': val_avg_loss_classify,
            'val_avg_loss_final': val_avg_loss_final
        }

        return metrics(label, pred), aux_info


    def predict(self, dataloader):
        if self.config['eval_mode']:
            print('month {} model loading...'.format(self.config['month']))
            self.model = ARGDModel(self.config)
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
