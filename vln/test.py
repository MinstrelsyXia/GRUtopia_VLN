def save_observations_special(self, camera_list:list, data_types:list, save_image_list=None, save_imgs=True, step_time=0):
            ''' Save observations from the agent
            '''
            obs = self.env.get_observations(data_type=data_types)
            DIR= os.path.expanduser("~/code/xxy/Matterport3D/data/grutopia_test")
            path_id = str(self.args.path_id)
            main_dir = os.path.join(DIR,self.args.env,path_id)
            if not os.path.exists(main_dir):
                os.makedirs(main_dir)
            camera_pose_dict = self.get_camera_pose()
            for camera in camera_list:
                cur_obs = obs[self.task_name][self.robot_name][camera]
                # save pose
                camera_pose=camera_pose_dict[camera]
                pos, quat = camera_pose[0], camera_pose[1]
                pose_save_path = os.path.join(main_dir,'poses.txt')
                with open(pose_save_path,'a') as f:
                    sep =""
                    f.write(f"{sep}{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}")
                    sep = "\n"
                
                # save rgb
                data_info = cur_obs['rgba'][...,:3]
                save_dir = os.path.join(main_dir,'rgb')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir, f"{camera}_{step_time}.png")
                try:
                    # plt.imsave(save_path, data_info)
                    cv2.imwrite(save_path, cv2.cvtColor(data_info, cv2.COLOR_RGB2BGR))
                    log.info(f"Images have been saved in {save_path}.")
                except Exception as e:
                    log.error(f"Error in saving camera rgb image: {e}")
                
                # save depth
                data_info = cur_obs['depth']
                save_dir = os.path.join(main_dir,'depth')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir, f"{camera}_{step_time}.npy")
                try:
                    np.save(save_path, data_info)
                    log.info(f"Depth have been saved in {save_path}.")
                except Exception as e:
                    log.error(f"Error in saving camera depth image: {e}")
                
                # save camera_intrinsic
                camera_params = cur_obs['camera_params']
                # 构造要保存的字典
                camera_info = {
                    "camera": camera,
                    "step_time": step_time,
                    "intrinsic_matrix": camera_params['cameraProjection'].tolist(),
                    "extrinsic_matrix": camera_params['cameraViewTransform'].tolist(),
                    "cameraAperture": camera_params['cameraAperture'].tolist(),
                    "cameraApertureOffset": camera_params['cameraApertureOffset'].tolist(),
                    "cameraFocalLength": camera_params['cameraFocalLength'],
                    "robot_init_pose": self.robot_init_pose
                }
                print(self.robot_init_pose)
                save_path = os.path.join(save_dir, 'camera_param.jsonl')

                # 将信息追加保存到 jsonl 文件
                with open(save_path, 'a') as f:
                    json.dump(camera_info, f)
                    f.write('\n')

                        
            return obs

    def get_camera_pose(self):
        '''
        Obtain position, orientation of the camera
        Output: position, orientation
        '''
        camera_dict = self.args.camera_list
        camera_pose = {}
        for camera in camera_dict:
            camera_pose[camera] = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].sensors[camera].get_world_pose()
        return camera_pose