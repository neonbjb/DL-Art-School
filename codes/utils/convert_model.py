# Tool that can be used to add a new layer into an existing model save file. Primarily useful for "progressive"
# models which can be trained piecemeal.

from utils import options as option
from models import create_model
import torch
import os

def get_model_for_opt_file(filename):
    opt = option.parse(filename, is_train=True)
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    return model, opt

def copy_state_dict_list(l_from, l_to):
    for i, v in enumerate(l_from):
        if isinstance(v, list):
            copy_state_dict_list(v, l_to[i])
        elif isinstance(v, dict):
            copy_state_dict(v, l_to[i])
        else:
            l_to[i] = v

def copy_state_dict(dict_from, dict_to):
    for k in dict_from.keys():
        if k == 'optimizers':
            for j in range(len(dict_from[k][0]['param_groups'])):
                for p in dict_to[k][0]['param_groups'][j]['params']:
                    del dict_to[k][0]['state']
                dict_to[k][0]['param_groups'][j] = dict_from[k][0]['param_groups'][j]
            dict_to[k][0]['state'].update(dict_from[k][0]['state'])
            print(len(dict_from[k][0].keys()), dict_from[k][0].keys())
            print(len(dict_to[k][0].keys()), dict_to[k][0].keys())
        assert k in dict_to.keys()
        if isinstance(dict_from[k], dict):
            copy_state_dict(dict_from[k], dict_to[k])
        elif isinstance(dict_from[k], list):
            copy_state_dict_list(dict_from[k], dict_to[k])
        else:
            dict_to[k] = dict_from[k]
    return dict_to

if __name__ == "__main__":
    os.chdir("..")
    model_from, opt_from = get_model_for_opt_file("../options/train_imgset_pixgan_progressive_srg2.yml")
    model_to, _ = get_model_for_opt_file("../options/train_imgset_pixgan_progressive_srg2_.yml")

    '''
    model_to.netG.module.update_for_step(1000000000000)
    l = torch.nn.MSELoss()
    o, _ = model_to.netG(torch.randn(1, 3, 64, 64))
    l(o, torch.randn_like(o)).backward()
    model_to.optimizer_G.step()
    o = model_to.netD(torch.randn(1, 3, 128, 128))
    l(o, torch.randn_like(o)).backward()
    model_to.optimizer_D.step()
    '''

    torch.save(copy_state_dict(model_from.netG.state_dict(), model_to.netG.state_dict()), "converted_g.pth")
    torch.save(copy_state_dict(model_from.netD.state_dict(), model_to.netD.state_dict()), "converted_d.pth")

    # Also convert the state.
    resume_state_from = torch.load(opt_from['path']['resume_state'])
    resume_state_to = model_to.save_training_state({}, return_state=True)
    resume_state_from['optimizers'][0]['param_groups'].append(resume_state_to['optimizers'][0]['param_groups'][-1])
    torch.save(resume_state_from, "converted_state.pth")







