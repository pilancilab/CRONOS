from optimizers import adamW, dadapt_adamW, sgd, shampoo, yogi, varpro, cronos_am, cronos_pro, adamW_nojit

# Setups specified optimizer
def get_optimizer(opt_params):
    
    lr = opt_params['lr']
    
    if opt_params['optimizer'] == 'SGD':
      momentum = opt_params['momentum']
      optimizer = sgd.SGD(lr, momentum)
    
    elif opt_params['optimizer'] == 'Adam':
      optimizer = adamW.AdamW(lr, 0)
    
    elif opt_params['optimizer'] == 'AdamW':
      gamma = opt_params['gamma']
      optimizer = adamW.AdamW(lr, gamma)
      
    elif opt_params['optimizer'] == 'AdamW_nojit':
      gamma = opt_params['gamma']
      optimizer = adamW_nojit.AdamW(lr, gamma) 
    
    
    elif opt_params['optimizer'] == 'DAdapt-AdamW':
      gamma = opt_params['gamma']
      optimizer = dadapt_adamW.Dadapt_AdamW(lr, gamma)
    
    elif opt_params['optimizer'] == 'Yogi':
      optimizer = yogi.Yogi(lr)

    elif opt_params['optimizer'] == 'Shampoo':
      optimizer = shampoo.Shampoo(lr) 
    
    elif opt_params['optimizer'] == 'VarPro':
      gamma = opt_params['gamma']
      optimizer = varpro.VarPro(lr, gamma)
    
    elif opt_params['optimizer'] == 'Cronos_AM':
      gamma = opt_params['gamma']
      optimizer = cronos_am.CronosAM(opt_params['cronos_params'], 
      lr, gamma)
    
    elif opt_params['optimizer'] == 'CronosPro':
      gamma = opt_params['gamma']
      optimizer = cronos_pro.CronosPro(opt_params['cronos_params'], 
      lr, gamma) 

    
    else:
      raise ValueError("The specified optimizer is not implemented")
    
    return optimizer