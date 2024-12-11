''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses

# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}

    return train


def GAN_training_function(G, D, GD, teacher, z_, y_, ema, state_dict, config, Class_reuglarizer=None):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])

        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                (D_fake, D_real), (D_adc_fake, D_adc_real), (D_ac_fake, D_ac_real), (D_mi_fake, D_mi_real), (
                D_am_fake, D_am_real), (D_fake_out_reg, D_real_out_reg), (fake_images, real_images) = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                           x[counter], y[counter], train_G=False,
                                           split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

                D_aux_loss = torch.tensor(0, device=D_loss.device)
                if config['loss'] == 'acgan':
                    D_ac_loss = losses.classifier_loss_dis(D_ac_real, y[counter], config['hinge'])
                    D_aux_loss = D_ac_loss
                elif config['loss'] == 'tacgan':
                    D_ac_loss = losses.classifier_loss_dis(D_ac_real, y[counter], config['hinge'])
                    D_mi_loss = losses.classifier_loss_dis(D_mi_fake, y_[:config['batch_size']], config['hinge'])
                    D_aux_loss = D_ac_loss + D_mi_loss
                elif config['loss'] == 'amgan':
                    D_loss = D_loss.detach()
                    D_aux_loss = losses.classifier_loss_dis(D_am_real, y[counter], config['hinge']) + \
                                 losses.classifier_loss_dis(D_am_fake, torch.ones_like(y_[:config['batch_size']]) *
                                                            utils.nclass_dict[config['dataset']], config['hinge'])
                elif config['loss'] == 'adcgan':
                    D_adc_loss_real = losses.classifier_loss_dis(D_adc_real, y[counter] * 2, config['hinge'])
                    D_adc_loss_fake = losses.classifier_loss_dis(D_adc_fake, y_[:config['batch_size']] * 2 + 1,
                                                                 config['hinge'])
                    D_aux_loss = D_adc_loss_real + D_adc_loss_fake
                D_aux_loss = config['D_lambda'] * D_aux_loss / float(config['num_D_accumulations'])


                with torch.no_grad():
                    latent_real = teacher(real_images, is_feat=True)
                    latent_fake = teacher(fake_images, is_feat=True)

                    D_loss_reg = losses.TripletLoss_reverse(latent_real, D_real_out_reg, D_fake_out_reg) + \
                                 losses.TripletLoss_reverse(latent_fake, D_fake_out_reg, D_real_out_reg)

                    D_loss_reg = config['D_reg_lambda'] * D_loss_reg / float(config['num_D_accumulations'])


                (D_loss + D_aux_loss + D_loss_reg).backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            D_fake, D_adc_fake, D_ac_fake, D_mi_fake, D_am_fake, imgs_gen = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])

            G_aux_loss = torch.tensor(0., device=G_loss.device)
            if config['loss'] == 'acgan':
                G_ac_loss = losses.classifier_loss_gen(D_ac_fake, y_, config['hinge'])
                G_aux_loss = G_ac_loss
            elif config['loss'] == 'tacgan':
                G_ac_loss = losses.classifier_loss_gen(D_ac_fake, y_, config['hinge'])
                G_mi_loss = losses.classifier_loss_gen(D_mi_fake, y_, config['hinge'])
                G_aux_loss = G_ac_loss - G_mi_loss
            elif config['loss'] == 'amgan':
                G_loss = G_loss.detach()
                G_aux_loss = losses.classifier_loss_gen(D_am_fake, y_, config['hinge'])
            elif config['loss'] == 'adcgan':
                G_adc_loss_pos = losses.classifier_loss_gen(D_adc_fake, y_ * 2, config['hinge'])
                G_adc_loss_neg = losses.classifier_loss_gen(D_adc_fake, y_ * 2 + 1, config['hinge'])
                G_aux_loss = G_adc_loss_pos - G_adc_loss_neg
            G_aux_loss = config['G_lambda'] * G_aux_loss / float(config['num_G_accumulations'])

            G_class_reg = Class_reuglarizer.loss(imgs_gen, teacher)
            G_class_reg = config['G_reg_lambda'] * G_class_reg / float(config['num_G_accumulations'])

            (G_loss + G_aux_loss + G_class_reg).backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])


        # shorten for small screen
        out = {'G': float(G_loss.item()),
               'DR': float(D_loss_real.item()),
               'DF': float(D_loss_fake.item()),
               'GA': float(G_aux_loss.item()),
               'DA': float(D_aux_loss.item()),
               'D_reg': float(D_loss_reg.item()),
               'G_reg': float(G_class_reg.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    return
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''


def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,
                                                 config['num_inception_images'],
                                                 num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
    state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID))
