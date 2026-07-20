import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import utils
import numpy as np
import os
import ipdb
from models.utils import *
from sklearn import metrics
import time 
import cv2
def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveVAD:
    def __init__(self, diffusion, args, config):
        super(DiffusiveVAD, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        
        if os.path.isfile(self.config.sampling.resume):
            self.diffusion.load_ddm_ckpt(self.config.sampling.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')



    def test(self, test_dataset, r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset)
        merge_flag = self.args.merge
        p_size = self.config.data.patch_size
        col_num = (256-64)//self.args.grid_r+1
        videos_list = test_dataset.video
        videos = test_dataset.videos
        labels_list = []
        label_length = 0
        img_list = {}
        patch_list = {}
        rec_list = {}
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers = 8,
        )
        labels = np.load('datasets/frame_labels_'+self.config.data.dataset+'.npy')
        for video in sorted(videos_list):
            video_name = video.replace("\\","/").split('/')[-1]
            if self.config.data.dataset not in ['shanghai','ub','xd','ucf']:
                labels_list = np.append(labels_list, labels[0][self.config.data.time_step+label_length:videos[video_name]['length']+label_length])
            else:
                labels_list = np.append(labels_list, labels[self.config.data.time_step+label_length:videos[video_name]['length']+label_length])
            label_length += videos[video_name]['length']
            img_list[video_name] = []
            patch_list[video_name] = []  
            rec_list[video_name] = []
        
        print('Evaluation of', self.config.data.dataset)
        print('----------------------------------------')
        for video in sorted(videos_list):
            video_name = video.replace("\\","/").split('/')[-1]
            img_list[video_name] = []  
            patch_list[video_name] = []  
            rec_list[video_name] = []

        cost_time = 0
        # 진단2: 픽셀별 오차 맵 저장 (env로 켬). ERRMAP_ONLY=쉼표목록이면 그 영상만.
        _dump_err = bool(os.environ.get("ERRMAP_DUMP"))
        _eo = os.environ.get("ERRMAP_ONLY")
        _err_only = set(_eo.split(",")) if _eo else None
        errmap_list = {}
        with torch.no_grad():
            for k,(x, video_names,c) in enumerate(tqdm((test_loader),desc="Testing Step")):  
                b = x.shape[0]
                x_cond = x[:, :self.config.data.time_step, :, :].to(self.diffusion.device)
                x_input = x[:, self.config.data.time_step:, :, :]
                x_dest = x[:, self.config.data.time_step:, :, :].to(self.diffusion.device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                x_output,closs,corners = self.diffusive_restoration(x_cond,x_dest, r=r, merge=merge_flag)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                cost_time= end-start + cost_time
                if merge_flag=="False":
                    x_output = inverse_data_transform(x_output).cpu().reshape(-1,b,3,64,64).permute(1,0,2,3,4)
                    corners_patch = torch.cat([x_input[:,:,:,x:x+p_size,y:y+p_size] for (x,y) in corners],dim=0)
                    corners_patch = corners_patch.reshape(-1,b,3,64,64).permute(1,0,2,3,4)
                    mse = (x_output-corners_patch).square()
                    mse = mse.mean(dim=(1, 2, 3)).max(1)[0]
                    for i,video in enumerate(video_names):
                        img_list[video].append(mse[i].cpu().detach().item()) 
                        patch_list[video].append(mse[i].cpu().detach().item()) 
                        rec_list[video].append(closs[i].cpu().detach().item()) 
                    if k%100==0:
                        utils.logging.save_image(x_output.reshape(-1,3,64,64), os.path.join(image_folder, f"output_{k}.png"),col_num,2)
                        utils.logging.save_image(corners_patch.reshape(-1,3,64,64), os.path.join(image_folder, f"input_{k}.png"),col_num,2)
                        utils.logging.save_image(((corners_patch-x_output)*2).reshape(-1,3,64,64), os.path.join(image_folder, f"res_{k}.png"),col_num,2)
                else:
                    x_output = inverse_data_transform(x_output).cpu()
                    mse = (x_output-x_input).square()
                    emap = mse.mean(dim=(1, 2))   # [b,H,W] 픽셀별 오차(채널·예측 평균)
                    mse_patch = [mse[:,:,:,x:x+p_size,y:y+p_size].mean(dim=(1, 2, 3, 4)).view(1,-1) for (x,y) in corners]
                    mse_patch = torch.cat(mse_patch).max(dim=0)[0]
                    mse = mse.mean(dim=(1, 2, 3, 4))
                    for i,video in enumerate(video_names):
                        img_list[video].append(mse[i].cpu().detach().item())
                        patch_list[video].append(mse_patch[i].cpu().detach().item())
                        rec_list[video].append(closs[i].cpu().detach().item())
                        if _dump_err and (_err_only is None or video in _err_only):
                            errmap_list.setdefault(video, []).append(
                                emap[i].cpu().numpy().astype("float32"))
                    if k%100==0:
                        utils.logging.save_image(x_output[:,0], os.path.join(image_folder, f"output_{k}.png"))
                        utils.logging.save_image(x_input[:,0], os.path.join(image_folder, f"input_{k}.png"))
                        utils.logging.save_image(((x_input-x_output)*2)[:,0], os.path.join(image_folder, f"res_{k}.png"))
        print("cost:{}".format(cost_time))
        print("fps:{}".format(test_dataset.__len__()/cost_time))
        if _dump_err and errmap_list:
            _ed = os.path.join("results/images", self.config.data.dataset, "errmaps")
            os.makedirs(_ed, exist_ok=True)
            for _v, _maps in errmap_list.items():
                np.save(os.path.join(_ed, _v + ".npy"), np.stack(_maps))
            print(f"[errmap] saved {len(errmap_list)} videos -> {_ed}")
        anomaly_score_total_list = []
        anomaly_score_total_ = []
        sum=0
        if not os.path.exists("logs/{}/pre_res/".format(self.config.data.dataset)):
            os.makedirs("logs/{}/pre_res/".format(self.config.data.dataset))
        aucs = []
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]   
            l=len(img_list[video_name])
            gt_i=labels_list[sum:sum+l] 
            pred_img = filt(img_list[video_name], range=38, mu=11)
            c_loss = filt(rec_list[video_name], range=38, mu=11)
            anomaly_score_total_list.append(score_sum(pred_img,c_loss,[1.,0.]))
            anomaly_score_total_.append(score_sum(pred_img,c_loss,[1.,0.]))
            lbl = np.array([0] + list(gt_i) + [1])
            pred = np.array([0] + list(pred_img) + [1])
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            res = metrics.auc(fpr, tpr)
            aucs.append(res)
            plt.figure(figsize=(6,3))
            plt.plot(gt_i)
            plt.plot(pred_img)
            plt.savefig("logs/{}/pre_res/{}.png".format(self.config.data.dataset,video_name))
            plt.close()
            sum=sum+l

            
         
        feature_dict = {}
        feature_dict['img_score']=img_list
        feature_dict['patch_score']=patch_list
        feature_dict['rec_score']=rec_list
        feature_dict['videos_list']=videos_list
        feature_dict['labels_list']=labels_list
        import pickle 
        with open('results/images/{}/{}.pkl'.format(self.config.data.dataset,self.config.data.dataset),'wb') as f:
            pickle.dump(feature_dict, f)  
        anomaly_score_total_list = np.concatenate(anomaly_score_total_list,axis=0)

        AUC_Socre = AUC(anomaly_score_total_list, np.expand_dims(labels_list, 0))
        print('The result of ', self.config.data.dataset)
        print('AUC: ', AUC_Socre*100, '%')

        ap_score = AP(anomaly_score_total_list, np.expand_dims(labels_list, 0))
        print('The result of ', self.config.data.dataset)
        print('AP: ', ap_score*100, '%')
        macro_auc = np.nanmean(aucs)
        print(f"MicroAUC: {AUC_Socre}, MacroAUC: {macro_auc}")


    # def colormap(self, g, p):
    #     import cv2
    #     mseimgs = ((p-g).square())[k,0].cpu().detach().numpy()
    #     mseimgs = mseimgs[:,:,np.newaxis]
    #     mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs)-np.min(mseimgs))
    #     mseimgs = mseimgs * 255
    #     mseimgs = mseimgs.astype(dtype=np.uint8)
    #     color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
    #     cv2.imwrite(os.path.join('MSE/MSE_{:04d}.jpg').format(k), color_mseimgs)

    def colormap(self, p, g, k):
        import cv2
        mseimgs = ((p-g).square())[k,0].cpu().detach().numpy()
        mseimgs = mseimgs[:,:,np.newaxis]
        mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs)-np.min(mseimgs))
        mseimgs = mseimgs * 255
        mseimgs = mseimgs.astype(dtype=np.uint8)
        color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('MSE/MSE_{:04d}.jpg').format(k), color_mseimgs)


    def restore(self, val_loader, r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset)
        p_size = self.config.data.patch_size
        merge_flag = self.args.merge
        col_num = (256-64)//self.args.grid_r+1
        with torch.no_grad():
            test_loader = iter(val_loader)
            print(len(test_loader))
            print(len(test_loader))
            
            for i in tqdm(range(4)):
                next(test_loader)
            i = 0
            
            # batch, cond = testdata.__getitem__(580)
            x, _, c = next(test_loader)
            b = x.shape[0]
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 6 else x
            x_cond = x[:, :self.config.data.time_step, :, :].to(self.diffusion.device)
            x_input = x[:, self.config.data.time_step:, :, :]
            # ipdb.set_trace()
            x_dest = x[:, -1:, :, :].to(self.diffusion.device)
            cost_time = 0
            
            start = time.time()
            x_output,closs,corners = self.diffusive_restoration(x_cond, x_dest, r=r, merge=merge_flag)
            end = time.time()
            cost_time = end - start + cost_time
            print("cost:{}".format(cost_time))
            print("fps:{}".format(16/cost_time))

            if merge_flag=="False":
                x_output = inverse_data_transform(x_output).cpu().reshape(-1,b,3,64,64).permute(1,0,2,3,4)
                # ipdb.set_trace()
                corners_patch = torch.cat([x_input[:,:,:,x:x+p_size,y:y+p_size] for (x,y) in corners],dim=0)
                corners_patch = corners_patch.reshape(-1,b,3,64,64).permute(1,0,2,3,4)
                mse = (x_output-corners_patch).square()
                # ipdb.set_trace()
                mse = mse.mean(dim=(1, 2, 3)).max(1)[0]
                
                print(mse)
                print(closs)
                utils.logging.save_image(x_output.reshape(-1,3,64,64), os.path.join(image_folder, f"output.png"),col_num,2)
                utils.logging.save_image(corners_patch.reshape(-1,3,64,64), os.path.join(image_folder, f"input.png"),col_num,2)
                utils.logging.save_image(((corners_patch-x_output)*2).reshape(-1,3,64,64), os.path.join(image_folder, f"res.png"),col_num,2)
            else:
                x_output = inverse_data_transform(x_output).cpu()
            
                mse = (x_output-x_input).square()
            
                mse = mse.mean(dim=(1, 2, 3, 4))
                ipdb.set_trace()
                print(mse)
                print(closs)
                utils.logging.save_image(x_output[:,0], os.path.join(image_folder, f"output.png"))
                utils.logging.save_image(x_input[:,0], os.path.join(image_folder, f"input.png"))
                utils.logging.save_image(((x_input-x_output)*2)[:,0], os.path.join(image_folder, f"res.png"))
    def restore_video(self, val_loader, r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset)
        p_size = self.config.data.patch_size
        merge_flag = self.args.merge
        col_num = (256-64)//self.args.grid_r+1
        cost_time = 0
        input_video = []
        out_video = []
        colormap = []
        with torch.no_grad():
            for k,(x, video_names,c) in enumerate(tqdm((val_loader),desc="Testing Step")):  
                b = x.shape[0]
                x_cond = x[:, :self.config.data.time_step, :, :].to(self.diffusion.device)
                x_input = x[:, self.config.data.time_step:, :, :]
                x_dest = x[:, self.config.data.time_step:, :, :].to(self.diffusion.device)
                
                start = time.time()
                x_output,closs,corners = self.diffusive_restoration(x_cond, x_dest, r=r, merge=merge_flag)
                end = time.time()
                cost_time = end - start + cost_time
                x_output = inverse_data_transform(x_output.squeeze()).cpu().permute(0,2,3,1).numpy()
                x_input = x_input.squeeze().permute(0,2,3,1).numpy()
                for i in range(len(x_output)):
                    inframe = (x_input[i] * 255).astype(np.uint8)
                    outframe= (x_output[i] * 255).astype(np.uint8)
                    mseimgs = cv2.absdiff(inframe,outframe)
                    # mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs)-np.min(mseimgs))
                    # mseimgs = (mseimgs * 255).astype(np.uint8)
                    input_video.append(inframe)
                    out_video.append(outframe)
                    colormap.append(mseimgs)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (808, 306))  # 调整宽度以适应间隔
        gap_color = (255, 255, 255)
        gap =  np.full((256, 20, 3), gap_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 0)  
        title_height = 50  
        titles = ['GT', 'MSE', 'Pre']
        title_frame = np.full((title_height, 808, 3), gap_color, dtype=np.uint8)
        width = 256
        for i, title in enumerate(titles):
            text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
            text_x = i * (width + 20) + (width - text_size[0]) // 2
            text_y = (title_height + text_size[1]) // 2
            cv2.putText(title_frame, title, (text_x, text_y), font, font_scale, text_color, font_thickness)
        print("length:{}".format(len(input_video)))
        for inputs, output, absdiff in zip(input_video, out_video, colormap):

            combined_frame = np.hstack((inputs, gap, absdiff, gap, output))[:,:,::-1]
            final_frame = np.vstack((title_frame, combined_frame))
            out.write(final_frame)
        out.release()
        print("output.mp4")
    def diffusive_restoration(self, x_cond, x_dest, r=None, merge=None):

        x_dest = 2 * x_dest - 1.0
        p_size = self.config.data.patch_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond[:,-1:].size(), device=self.diffusion.device)
        
        x_output,closs = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size, merge=merge)

        return x_output,closs,corners

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, _, c, h, w = x_cond.shape
       
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]

        return h_list, w_list
