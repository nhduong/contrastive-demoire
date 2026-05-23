from utils import *

@hydra.main(version_base=None, config_path="configs", config_name="hydra_config")
def main(cfg: DictConfig):
    def main_task(args):
        # if args.detect_anomaly:
        #     torch.autograd.set_detect_anomaly(True)
        
        # set seed
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
        
        # Hunggingface accelerate
        if args.utils.hf_accelerate:
            # ddp_kwargs = DistributedDataParallelKwargs()
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False) # for BatchNorm
            accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
            device = accelerator.device
            if args.seed is not None:
                accelerator.wait_for_everyone()
                accelerate.utils.set_seed(args.seed)
        else:
            accelerator = None
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # global variables for logs
        global best_psnr, best_ssim, best_epoch
        best_psnr = 0
        best_ssim = 0
        best_epoch = -1
        lr_cycle = 1

        # switch to validation mode?
        if args.exp.evaluate:
            args.exp.fast_psnr = False
            args.data.val.patch_size = None

        # logs + tensorboard
        accelerator.wait_for_everyone()
        time_id, _ = get_current_time()
        args.time_id = time_id
        log_dir         = "./outputs"
        logs_path       = os.path.join(os.path.join(log_dir, "tb"), "train")
        logs_path_val   = os.path.join(os.path.join(log_dir, "tb"), "val")
        
        # backup code and configs
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(logs_path, exist_ok=True)
            os.makedirs(logs_path_val, exist_ok=True)
            os.makedirs(os.path.join(log_dir, "cp"), exist_ok=True)

            subprocess.run("cp " + get_original_cwd() + "/main.py " + log_dir, shell=True)
            subprocess.run("cp " + get_original_cwd() + "/utils.py " + log_dir, shell=True)
            os.makedirs(log_dir + "/configs", exist_ok=True)
            os.makedirs(log_dir + "/models", exist_ok=True)
            subprocess.run("cp " + get_original_cwd() + "/configs/*_config.yaml " + log_dir + "/configs/", shell=True)
            subprocess.run("cp " + get_original_cwd() + "/" + args.models.path.replace(".", "/") + ".py " + log_dir + "/" + args.models.path.replace(".", "/") + ".py", shell=True)

        # tensorboard writers
        if args.utils.tb and accelerator.is_main_process:
            writer = SummaryWriter(log_dir=logs_path, flush_secs=1)
            writer_val = SummaryWriter(log_dir=logs_path_val, flush_secs=1)
            print(">>> Tensorboard logs saved to: {}".format(logs_path))
            print(">>> Tensorboard logs saved to: {}".format(logs_path_val))
        else:
            writer = None
            writer_val = None
            print(">>> Not using Tensorboard")

        # logs to file
        accelerator.wait_for_everyone()
        log_fn = os.path.join(log_dir, "proc_" + str(accelerator.process_index) + "_" + args.exp.name + "_" + args.data.name + "_" + args.exp.note + "_" + time_id + ".log")
        if args.utils.log2file:
            sys.stdout = open(log_fn, "w")
            sys.stderr = sys.stdout
            logging.basicConfig(filename=log_fn, filemode="a")
            print(">>> Logs saved to: {}".format(log_fn))
            
            # flush the logs
            sys.stdout.flush()
        
        if args.utils.tb and accelerator.is_main_process:
            writer_val.add_text("start_time", time_id, 0)
            writer_val.flush()
        
        # starting time
        t0 = datetime.datetime.now()
        
        proc_id = os.getpid()
        print(">>> proccess ID:", proc_id)
        print(">>> number of GPUs:", accelerator.num_processes)
        print(">>> precision type:", accelerator.mixed_precision)
        print(">>> distributed type:", accelerator.distributed_type)
        print(">>> is bf16 available: ", accelerate.utils.is_bf16_available())

        print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

        # adding logs
        args.init_date = "[" + time_id + "]"
        args.proc_id = proc_id
        args.app = "[" + args.exp.name + "_" + args.data.name + "]_[" + args.exp.note + "]"
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        args.node = platform.node()
        args.log_path = os.getcwd() # log_dir
        print("===========================")
        print(args)
        print("===========================")
        log_system_info()
        print("===========================")
        sys.stdout.flush()
        
        # create demo_net
        print("=> creating demo_net (%s)..." % args.models.main._target_)
        demo_net = instantiate(args.models.main)
        demo_net._initialize_weights()
                
        # vgg block for perceptual loss
        vgg_blocks = []
        vgg_blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        vgg_blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        vgg_blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        vgg_blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # freeze vgg blocks
        for bl in vgg_blocks:
            for p in bl:
                p.requires_grad = False
        vgg_blocks = torch.nn.ModuleList(vgg_blocks)

        # print model size
        if accelerator.is_main_process:
            print(">>> number of params (demoireing net): {:,}".format(count_parameters(demo_net)))

            # flops count
            if args.exp.calc_flops:
                print(">>> calculating FLOPs (demoireing net) for an input of size (1, 3, 1088, 1920)... ", end="")
                demo_net.eval()
                with torch.no_grad():
                    input = torch.randn(1, 3, 1088, 1920)
                    
                    # padding the image
                    _, _, h_1, w_1 = input.size()

                    # pad image such that the resolution is a multiple of 32 for multi-scale processing
                    w_pad_1 = (math.ceil(w_1/32)*32 - w_1) // 2
                    h_pad_1 = (math.ceil(h_1/32)*32 - h_1) // 2
                    input = img_pad(input, w_r=w_pad_1, h_r=h_pad_1)
                    
                    flops = FlopCountAnalysis(demo_net, input)
                    print(">>> FLOPs (demoireing net): {:,}T".format(flops.total() * 1e-12))
                demo_net.train()
                
            # # memory usage estimation
            # input = torch.randn(1, 3, 1088, 1920).cuda()
            # demo_net = demo_net.cuda()
            # demo_net.eval()
            # with torch.no_grad():
            #     demo_net(input)
                
            # # calculate GPU memory usage
            # time.sleep(10)
            
            # sys.stdout.flush()
            # # TAKE A LOOK AT the nvidia-smi output to see the estimated memory usage of the model!

            # # tensorboard demo_net graph
            # img_size = 256
            # x = torch.randn(1, 3, img_size, img_size)

            # if args.utils.tb:
            #     print(">>> logging demo_net graph to Tensorboard...")
            #     writer_val.add_graph(demo_net, x)
            #     writer_val.flush()

        # loss functions
        criterion_1 = instantiate(args.losses.l1.func)
        criterion_2 = instantiate(args.losses.perceptual.func, blocks=vgg_blocks)

        # optimizer
        optimizer = instantiate(args.opt.alg,
                                params=demo_net.parameters(),
                                lr=args.opt.lr, betas=(args.opt.beta1, args.opt.beta2)
        )
        
        print(optimizer)
        print(">>> lr: {}".format(optimizer.param_groups[0]["lr"]))
        
        # learning rate scheduler
        scheduler = instantiate(args.schedule.cosine.alg, optimizer=optimizer, T_0=args.opt.T_0, eta_min=args.opt.eta_min, verbose=True)

        # print(scheduler)
        lr = optimizer.param_groups[0]["lr"]

        # pytorch data loaders
        print("=> creating data loaders...")
        if args.data.val.patch_size:
            print("\n=====================\n>>> WARNING: using VAL_PATCH_SIZE for fast validation\n\tfor ACCURATE results, use full images\n=====================\n")
        
        train_dataset = MoireDataset(
            data_path=args.data.path,
            sub_dir=args.data.train.path,
            moire_dir=args.data.moire_dir, clean_dir=args.data.clean_dir,
            data_name=args.data.name,
            is_training=True,
            try_it=args.utils.try_it,
            patch_size=args.data.train.patch_size,
        )

        val_dataset = MoireDataset(
            data_path=args.data.path,
            sub_dir=args.data.val.path,
            moire_dir=args.data.moire_dir, clean_dir=args.data.clean_dir,
            data_name=args.data.name,
            is_training=False,
            try_it=args.utils.try_it,
            patch_size=args.data.val.patch_size,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.data.train.batch_size, shuffle=True,
            num_workers=args.opt.workers, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.data.val.batch_size, shuffle=args.utils.try_it,
            num_workers=args.opt.workers, pin_memory=True)
        
        if accelerator.is_main_process:
            print("args.data.train.batch_size", args.data.train.batch_size)
            print("args.data.train.patch_size", args.data.train.patch_size)
            print("args.data.val.batch_size", args.data.val.batch_size)
            
        # Hunggingface accelerate preparation
        if args.utils.hf_accelerate:
            demo_net, optimizer, criterion_1, criterion_2, train_loader, val_loader = accelerator.prepare(demo_net, optimizer, criterion_1, criterion_2, train_loader, val_loader)

        # optionally resume from a checkpoint
        has_resumed = False
        if args.exp.resume:
            best_psnr_now = 0
            best_epoch_now = 0
            cor_ssim_now = 0
            if len(args.exp.evaluate_epochs) > 0:
                # if checkpoint directory is not ready, wait for it
                if args.exp.evaluate_wait_for_epochs:
                    while not os.path.isdir(args.exp.resume):
                        print(">>> waiting for checkpoints in: {}".format(args.exp.resume))
                        time.sleep(60 * 3)

                if os.path.isdir(args.exp.resume):
                    for epoch_id in args.exp.evaluate_epochs:
                        # clean up PyTorch memory
                        torch.cuda.empty_cache()

                        resume_epoch_1 = os.path.join(args.exp.resume, "{:04d}_BEST_cp_dir".format(epoch_id))
                        resume_epoch_2 = os.path.join(args.exp.resume, "{:04d}_cp_dir".format(epoch_id))

                        # if checkpoint directory is not ready, wait for it
                        while not os.path.isdir(resume_epoch_1) and not os.path.isdir(resume_epoch_2):
                            if args.exp.evaluate_wait_for_epochs:
                                print(">>> waiting for checkpoint: {}".format(epoch_id))
                                time.sleep(60 * 3)

                        # checkpoint directory containing the models, optimizers, etc.
                        if os.path.isdir(resume_epoch_1):
                            resume_epoch = resume_epoch_1
                        elif os.path.isdir(resume_epoch_2):
                            resume_epoch = resume_epoch_2

                        # if checkpoint directory is not ready, wait for it
                        if args.exp.evaluate_wait_for_epochs:
                            while not os.path.isdir(resume_epoch):
                                print(">>> waiting for checkpoint: {}".format(resume_epoch))
                                time.sleep(60 * 3)
                        
                        # load the checkpoint
                        if os.path.isdir(resume_epoch):
                            print("=> loading checkpoint '{}'".format(resume_epoch))
                            try:
                                accelerator.load_state(resume_epoch)
                            except Exception as e:
                                print(">>> accelerator.load_state failed, trying to load model weights only")
                                # load model weights
                                state_dict = load_file(os.path.join(resume_epoch, "model.safetensors"))
                                accelerator.unwrap_model(demo_net).load_state_dict(state_dict)
                                                        
                            if args.opt.capturable:
                                optimizer.param_groups[0]['capturable'] = True
                            else:
                                optimizer.param_groups[0]['capturable'] = False

                            # usefull variables
                            try:
                                checkpoint = torch.load(os.path.join(resume_epoch, "info.pth.tar"), map_location = lambda storage, loc: storage.cuda())
                            except Exception as e:
                                pass

                            try:
                                args.opt.start_epoch = checkpoint['epoch'] + 1
                                best_psnr = checkpoint['best_psnr']
                                best_ssim = checkpoint['cor_ssim']
                                best_epoch = checkpoint['best_epoch']
                            except Exception as e:
                                # reset the best values???
                                print(e)
                            
                            try:
                                lr_cycle = checkpoint['lr_cycle']
                            except:
                                print(">>> lr_cycle not loaded")
                                pass

                            print(">>> start epoch: {}".format(args.opt.start_epoch))
                            print(">>> best psnr: {}".format(best_psnr))
                            print(">>> best ssim: {}".format(best_ssim))
                            print(">>> best epoch: {}".format(best_epoch))
                            
                            print("=> loaded checkpoint '{}' (epoch {})".format(resume_epoch, args.opt.start_epoch))
                            
                            has_resumed = True

                            if args.exp.evaluate:
                                # an image file
                                if os.path.isfile(args.data.val.file):
                                    img = cv2.imread(args.data.val.file)
                                    cv2.imwrite("input.png", img)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = img.astype(np.float32) / 255.0
                                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

                                    # pad image such that the resolution is a multiple of 32 for multi-scale processing
                                    _, _, h, w = img.size()
                                    w_pad = (math.ceil(w/32)*32 - w) // 2
                                    h_pad = (math.ceil(h/32)*32 - h) // 2
                                    img = img_pad(img, w_r=w_pad, h_r=h_pad)
                                    # calculating the demo_net output
                                    demo_net.eval()
                                    with torch.no_grad():
                                        C_1, _, _, _, _, _ = demo_net(img)
                                        
                                        # saving the output
                                        C_1 = C_1.squeeze(0).permute(1, 2, 0).cpu().numpy()
                                        C_1 = np.clip(C_1, 0, 1)
                                        C_1 = cv2.cvtColor(C_1, cv2.COLOR_RGB2BGR)
                                        cv2.imwrite("output.png", C_1 * 255.0)
                                # dataset instead
                                else:
                                    # evaluation metrics
                                    if args.exp.fast_psnr:
                                        print("\n=====================\n>>> WARNING: using FAST_PSNR for fast validation\n\tfor ACCURATE results, modify fast_psnr in the config file\n=====================\n")
                                    compute_metrics = create_metrics(args, device=device, accelerator=accelerator)
                                    
                                    # validation
                                    val_psnr, val_ssim = validate(val_loader, demo_net, device, accelerator, args.opt.start_epoch - 1, args, compute_metrics, t0)

                                    if val_psnr > 0 or val_ssim > 0:
                                        if args.utils.tb and accelerator.is_main_process:
                                            writer_val.add_scalar(args.data.name + '/met/psnr', val_psnr, args.opt.start_epoch - 1)
                                            writer_val.add_scalar(args.data.name + '/met/ssim', val_ssim, args.opt.start_epoch - 1)
                                            writer_val.flush()
                                    
                                    # check if the current checkpoint is the best
                                    if val_psnr > best_psnr_now:
                                        best_psnr_now = val_psnr
                                        cor_ssim_now = val_ssim
                                        best_epoch_now = args.opt.start_epoch - 1
                                        print("*** best psnr now: {}".format(best_psnr_now))
                                        print("*** best ssim now: {}".format(cor_ssim_now))
                                        print("*** best epoch now: {}".format(best_epoch_now))
                        else:
                            print("=> no checkpoint found at '{}'".format(resume_epoch))
                else:
                    print('>>> checkpoint directory not found. exiting...')
            else:
                print('>>> specify evaluate_epochs in config file. exiting...')

        # Hunggingface accelerate preparation
        if args.utils.hf_accelerate:
            vgg_blocks = accelerator.prepare(vgg_blocks)
                
        # terminate if evaluation only
        if args.exp.evaluate:
            if args.utils.tb and accelerator.is_main_process:
                writer_val.add_text("finished_time", get_current_time()[0], 0)
                writer_val.flush()
                writer_val.close()
                writer.close()

            print("testing finished!")
            return
        # ---------------------------------------------
        # For testing, the program ends here!
        # ----------------------------------------------



        # ----------------------------------------------
        # For training/resuming, the program continues
        # ----------------------------------------------
        # about to start training
        if args.opt.forced_start_epoch > -1:
            args.opt.start_epoch = args.opt.forced_start_epoch
        epoch = args.opt.start_epoch

        # evaluation metrics
        if args.exp.fast_psnr:
            print("\n=====================\n>>> WARNING: using FAST_PSNR for fast validation\n\tfor ACCURATE results, modify fast_psnr in the config file\n=====================\n")
        compute_metrics = create_metrics(args, device=device, accelerator=accelerator)
        compute_metrics = accelerator.prepare(compute_metrics)

        args.exp.evaluate = args.exp.calc_mets

        print("\n===========================\n")
        print(">>> training with the following losses:")
        if args.losses.l1.use:
            print(">>> using L1 loss with weight: {}".format(args.losses.l1.weight))
        if args.losses.perceptual.use:
            print(">>> using Perceptual loss with weight: {}".format(args.losses.perceptual.weight))
        if args.losses.contrastive.use:
            print(">>> using Contrastive loss with weight: {}".format(args.losses.contrastive.weight))
        print("\n===========================\n")

        # initialize variables
        val_psnr = val_ssim = 0        

        print("\n===========================\n")
        # print args items
        for arg in args:
            print(arg, getattr(args, arg))
        print("\n===========================\n")
        
        if has_resumed:
            print(">>> resuming training from epoch: {}".format(args.opt.start_epoch))
            print(">>> updating lr...")
            print(">>> lr (BEFORE): {}".format(optimizer.param_groups[0]["lr"]))
            scheduler.step(epoch % args.opt.T_0)
            print(">>> lr (AFTER): {}".format(optimizer.param_groups[0]["lr"]))

        # start training
        try:
            while True:
                # end of training
                if epoch >= args.opt.epochs:
                    break

                # reset lr
                if epoch % args.opt.T_0 == 0: # epoch 0, 50, 100, ...
                    print("===========================")
                    print(">>> resetting lr...")
                    print("===========================")
                    for ggg in optimizer.param_groups:
                        ggg['lr'] = args.opt.lr
                    
                    scheduler = instantiate(args.schedule.cosine.alg, optimizer=optimizer, T_0=args.opt.T_0, eta_min=args.opt.eta_min, verbose=True)
                    print(">>> lr: {}".format(optimizer.param_groups[0]["lr"]))
                    print(scheduler)
                
                # minimize lr at the end of the cycle
                if epoch % args.opt.T_0 == args.opt.T_0 - 1: # epoch 49, 99, 149, ...
                    print("===========================")
                    print(">>> minimizing lr...")
                    print("===========================")
                    for ggg in optimizer.param_groups:
                        ggg['lr'] = args.opt.eta_min
                    print(">>> lr: {}".format(optimizer.param_groups[0]["lr"]))

                lr = optimizer.param_groups[0]["lr"]

                if accelerator.is_main_process:
                    if args.utils.tb:
                        writer_val.add_scalar('z/lr', lr, epoch)
                        writer_val.flush()

                # training
                train_total_loss, train_l1_loss, train_per_loss, train_cont_loss, train_psnr, train_ssim = \
                    train(train_loader, demo_net, accelerator, device, criterion_1, criterion_2, optimizer, epoch, args, t0, lr, compute_metrics)
                
                # loss logging
                if accelerator.is_main_process:
                    if args.utils.tb:
                        writer.add_scalar(args.data.name + '/loss/total', get_tensor_value(train_total_loss), epoch)
                        if args.losses.l1.use:
                            writer.add_scalar(args.data.name + '/loss/l1', get_tensor_value(train_l1_loss), epoch)
                        if args.losses.perceptual.use:
                            writer.add_scalar(args.data.name + '/loss/perceptual', get_tensor_value(train_per_loss), epoch)
                        if args.losses.contrastive.use:
                            writer.add_scalar(args.data.name + '/loss/contrastive', get_tensor_value(train_cont_loss), epoch)
                        if train_psnr > 0 or train_ssim > 0:
                            writer.add_scalar(args.data.name + '/met/psnr', get_tensor_value(train_psnr), epoch)
                            writer.add_scalar(args.data.name + '/met/ssim', get_tensor_value(train_ssim), epoch)
                        writer.flush()

                # validation
                if not args.exp.dont_calc_mets_at_all and not args.exp.train_only:
                    if (epoch + 1) % round(args.opt.T_0 / 2) == 0 or epoch % args.opt.T_0 == 0 or args.exp.evaluate:
                        val_psnr, val_ssim = validate(val_loader, demo_net, device, accelerator, epoch, args, compute_metrics, t0)

                # best epoch?
                accelerator.wait_for_everyone()
                if val_psnr >= best_psnr or val_ssim >= best_ssim:
                    is_best = True
                    best_epoch = epoch
                    best_psnr = val_psnr
                    best_ssim = val_ssim
                    
                    # text logging
                    if accelerator.is_main_process:
                        if args.utils.tb:
                            writer_val.add_text('best', 'Best val PSNR {0} | val SSIM {1} @ epoch {2}'.format(best_psnr, best_ssim, best_epoch), 0)
                            writer_val.flush()
                else:
                    is_best = False
                    
                # loss logging
                if accelerator.is_main_process:
                    if args.utils.tb:
                        if val_psnr > 0 or val_ssim > 0:
                            writer_val.add_scalar(args.data.name + '/met/psnr', val_psnr, epoch)
                            writer_val.add_scalar(args.data.name + '/met/ssim', val_ssim, epoch)
                        writer_val.flush()

                # save BEST checkpoints
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if args.save_cp:
                        if is_best or not args.exp.calc_mets:
                            cp_dir = os.path.join(log_dir, "cp", "{:04d}_BEST_cp_dir".format(epoch))
                        else:
                            cp_dir = os.path.join(log_dir, "cp", "{:04d}_cp_dir".format(epoch))

                        print("=> saving checkpoint '{}'".format(epoch), end=" ")
                        # models, optimizer, scheduler, and any stateful objects
                        accelerator.save_state(output_dir=cp_dir)
                        # usefull variables
                        state = {
                            'epoch': epoch,
                            'best_psnr': best_psnr,
                            'cor_ssim': best_ssim,
                            'best_epoch': best_epoch,
                            'lr': lr,
                            'lr_cycle' : lr_cycle,
                        }
                        torch.save(state, os.path.join(cp_dir, "info.pth.tar"))
                        print("done!")
                        
                    print(
                        '## train PSNR {} | train SSIM {} @ epoch {}\n'
                        '>> Best val PSNR {} | val SSIM {} @ epoch {}\n'
                        .format(train_psnr, train_ssim, epoch, best_psnr, best_ssim, best_epoch)
                    )

                # next epoch
                epoch += 1
                
                # learning rate scheduling
                print(">>> lr (BEFORE): {}".format(optimizer.param_groups[0]["lr"]))
                scheduler.step(epoch % args.opt.T_0)
                print(">>> lr (AFTER): {}".format(optimizer.param_groups[0]["lr"]))

            if accelerator.is_main_process:
                print("finished!")

                if args.utils.tb:
                    writer_val.add_text("finished_time", get_current_time()[0], 0)
                    writer_val.flush()
                    writer_val.close()
                    writer.close()

        except Exception as e:
            logging.error(traceback.format_exc())
                    
    main_task(cfg)


if __name__ == '__main__':
    main()
