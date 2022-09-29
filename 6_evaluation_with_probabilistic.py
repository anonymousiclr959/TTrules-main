import argparse
import json
import os
from copy import copy, deepcopy
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from config.config import Config, two_args_str_float, transform_input_filters2, transform_input_thr, \
    transform_input_filters_multiple
from config.config import str2bool, two_args_str_int, str2list, \
    transform_input_filters, transform_input_lr
from ttnet.eval import load_model, evalmodel_3, eval_fairness_model, load_data_train, opportunity, dp, load_data_modt
from ttnet.load_datasets import read_csv, DBEncoder
from ttnet.sat_inference_normal import get_mapping_filter, load_cnf_dnf, inference_layer_by_layer_scale

config_general = Config(path="config/")
if config_general.dataset == 'adult':
    config = Config(path='config/adult/')
    continuous_dict = {1: "age_ST_", 3: "poids_binaire_raciste_ST_",
                       5: "years_of_education_ST_", 11: "capital_gain_ST_",
                       12: "capital_loss_ST_", 13: "hours_per_week_ST_"}
elif config_general.dataset == "law":
    config = Config(path="config/law/")
    continuous_dict = {}
elif config_general.dataset == "compas":
    config = Config(path="config/compas/")
    continuous_dict = {1: 'age_ST_', 4: 'diff_custody_ST_', 5: 'diff_jail_ST_', 6: 'priors_count_ST_'}
elif config_general.dataset == "cancer":
    config = Config(path="config/cancer/")
    continuous_dict = {}
elif config_general.dataset == "diabetes":
    config = Config(path="config/diabetes/")
    continuous_dict = {7: 'time_in_hospital_ST_', 9: 'num_lab_procedures_ST_', 10: 'num_procedures_ST_',
                       11: 'num_medications_ST_', 15: 'number_diagnoses_ST_'}
elif config_general.dataset == "allstate":
    config = Config(path="config/allstate/")
    continuous_dict = {2: 'time_after_', 9: 'age_oldest_ST_', 10: 'age_youngest_ST_',
                       13: 'duration_previous_ST_', 14: 'cost_ST_'}
else:
    raise 'PB'

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=config_general.dataset)
parser.add_argument("--kfold", default=config.general.kfold, type=two_args_str_int)
parser.add_argument("--checkpoint", default=config.general.checkpoint)
parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, choices=["cuda", "cpu"])
parser.add_argument("--device_ids", default=config.general.device_ids, type=str2list)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--DATA_DIR", default=config.general.DATA_DIR)
parser.add_argument("--num_workers", default=config.general.num_workers, type=int)
parser.add_argument("--sensitive_fairness", default=config.general.sensitive_fairness, type=str2list)

parser.add_argument("--quant_step", default=config.model.quant_step, type=two_args_str_float)
parser.add_argument("--famille", default=config.model.famille)
parser.add_argument("--cbd", default=config.model.cbd)

parser.add_argument("--type_blocks", default=config.model.type_blocks, type=transform_input_filters2)
parser.add_argument("--last_layer", default=config.model.last_layer,
                    choices=["float", "bin", "binpos", "continu_bin", "continu_binpos"])
parser.add_argument("--Blocks_filters_output", default=config.model.Blocks_filters_output, type=transform_input_filters)
parser.add_argument("--Blocks_amplifications", default=config.model.Blocks_amplifications, type=transform_input_filters)
parser.add_argument("--Blocks_strides", default=config.model.Blocks_strides, type=transform_input_filters)
parser.add_argument("--type_first_layer_block", default=config.model.type_first_layer_block, choices=["float", "bin"])
parser.add_argument("--kernel_size_per_block", default=config.model.kernel_size_per_block, type=transform_input_filters)
parser.add_argument("--groups_per_block", default=config.model.groups_per_block, type=transform_input_filters)
parser.add_argument("--padding_per_block", default=config.model.padding_per_block, type=transform_input_filters)
parser.add_argument("--thr_bin_act", default=config.model.thr_bin_act, type=transform_input_thr)
parser.add_argument("--random_permut", default=config.model.random_permut, type=str2bool)
parser.add_argument("--load_permut", default=config.model.load_permut)
parser.add_argument("--repeat_permut", default=config.model.repeat_permut, type=two_args_str_int)

parser.add_argument("--thr_bin_act_test", default=config.eval.thr_bin_act_test, type=transform_input_thr)

parser.add_argument("--adv_epsilon", default=config.train.adv_epsilon)
parser.add_argument("--adv_step", default=config.train.adv_step)
parser.add_argument("--n_epoch", default=config.train.n_epoch, type=two_args_str_int)
parser.add_argument("--lr", default=config.train.lr, type=transform_input_lr)
parser.add_argument("--data_augmentation", default=config.train.data_augmentation)

parser.add_argument("--batch_size_test", default=config.eval.batch_size_test, type=two_args_str_int)
parser.add_argument("--pruning", default=config.eval.pruning, type=str2bool)
parser.add_argument("--coef_mul", default=config.eval.coef_mul, type=two_args_str_int)

parser.add_argument("--attack_eps", default=config.verify.attack_eps, type=two_args_str_float)
parser.add_argument("--path_load_model", default=config.eval.path_load_model, type=two_args_str_int)

parser.add_argument("--dc_logic", default=config.get_TT.dc_logic, type=str2bool)
parser.add_argument("--filtrage", default=config.get_TT.filtrage, type=two_args_str_int)
# parser.add_argument("--continuous_dict", default=config.get_TT.continuous_dict, type=dict)

parser.add_argument("--load_general", default=config.verify.load_general, type=str2bool)
parser.add_argument("--load_specific", default=config.verify.load_specific)

parser.add_argument("--load_general_exception_filters", default=config.verify.load_general_exception_filters, type=str2bool)
parser.add_argument("--filters_correlation_pos", default=config.verify.filters_correlation_pos, type=transform_input_filters_multiple)
parser.add_argument("--filters_correlation_neg", default=config.verify.filters_correlation_neg, type=transform_input_filters_multiple)
parser.add_argument("--fairnessornot", default=config.verify.fairnessornot, type=str2bool)


args = parser.parse_args()
args.path_save_model = args.path_load_model + "/"
device = torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")
args.continuous_dict = continuous_dict

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.filters_correlation_pos = transform_input_filters_multiple(args.filters_correlation_pos)
args.filters_correlation_neg = transform_input_filters_multiple(args.filters_correlation_neg)

args.json_res_all = {}

args.json_res_all["dc_logic"] = args.dc_logic
args.json_res_all["filtrage"] = args.filtrage
args.json_res_all["repeat_permut"] = args.repeat_permut
args.json_res_all["random_permut"] = args.random_permut
args.json_res_all["load_general_exception_filters"] = args.load_general_exception_filters
args.json_res_all["accuracy"] = 0
args.json_res_all["DP"] = 0
args.json_res_all["OP"] = 0
args.json_res_all["f1"] = 0
args.json_res_all["precision"] = 0
args.json_res_all["recall"] = 0
args.json_res_all["confusion_matrix"] = 0
args.json_res_all["Number_rules"] = 0
args.json_res_all["Number_conditions"] = 0
args.json_res_all["Average_conditions_per_rule"] = 0
args.json_res_all["Std_conditions_per_rule"] = 0
args.json_res_all["Number_correlation_POS"] = 0
args.json_res_all["Number_correlation_NEG"] = 0
args.json_res_all["fairnes_true"] = args.fairnessornot
print(args)

quantized_model_train, dataloaders, testset = load_model(args, args.path_save_model, device)
print(quantized_model_train, quantized_model_train.feature_start)
feature_start = quantized_model_train.feature_start
for index_th, value_th in enumerate(args.thr_bin_act_test):
    Tici = value_th
    if index_th == 0:
        quantized_model_train.features[feature_start + 1].T = Tici
    else:
        quantized_model_train.features[feature_start + 1 + index_th].act.T = Tici

# final_mask_noise
#acc_ref, final_mask_noise = evalmodel_3(args, quantized_model_train, dataloaders, device, val_phase=['val'])
#_, _, _, _, _, _ = eval_fairness_model(args.mappings, args.sensitive_fairness, quantized_model_train, dataloaders, device)
img = testset[0][0].unsqueeze(0).to(device)
image = testset[0][0].unsqueeze(0).to(device)
del dataloaders
unfold_all = {}
for numblockici in range(len(args.type_blocks)):
    if args.type_blocks[numblockici] == "multihead_TTblock":
        unfold_all[numblockici] = [torch.nn.Unfold(kernel_size=args.kernel_size_per_block_multihead[0],
                                                   stride=args.Blocks_strides[numblockici],
                                                   padding=args.paddings_per_block_multihead[0]),
                                   torch.nn.Unfold(kernel_size=args.kernel_size_per_block_multihead[1],
                                                   stride=args.Blocks_strides[numblockici],
                                                   padding=args.paddings_per_block_multihead[1]),
                                   torch.nn.Unfold(kernel_size=args.kernel_size_per_block_multihead[2],
                                                   stride=args.Blocks_strides[numblockici],
                                                   padding=args.paddings_per_block_multihead[2])]
    else:
        unfold_all[numblockici] = [
            torch.nn.Unfold(kernel_size=(args.kernel_size_per_block[numblockici], 1),
                            stride=args.Blocks_strides[numblockici],
                            padding=args.padding_per_block[numblockici])]

# mapping filter


data_path = os.path.join(args.DATA_DIR, args.dataset + '.data')
info_path = os.path.join(args.DATA_DIR, args.dataset + '.info')
X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
db_enc = DBEncoder(f_df, discrete=False)
db_enc.fit(X_df, y_df)
X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
if args.random_permut is False:
    # if os.path.exists(os.path.join(args.load_permut, dataset + "_original_mapping.json")):
    print("Loading existing mapping")
    print(args.load_permut)
    with open(os.path.join(args.load_permut), 'r') as jfile:
        mappings = json.load(jfile)
else:
    with open(os.path.join(args.path_save_model, args.dataset + "_mapping.json"), 'r') as jfile:
        mappings = json.load(jfile)
X = db_enc.change_features_order(X, mappings)

# print(db_enc.X_fname)
fname_encode = {i: val for i, val in enumerate(db_enc.X_fname)}
# print(fname_encode)
fname = torch.Tensor(list(fname_encode.keys())).unsqueeze(0).unsqueeze(0)
# print(fname)
del X_df, y_df, f_df, label_pos, X, y

unfold_name = unfold_all[0][0]
# print(fname.shape)
# print(unfold_name)
input_vu_par_cnn_avant_unfold = fname[:, :, :].unsqueeze(-1)
input_vu_par_cnn_et_sat_ref = unfold_name(input_vu_par_cnn_avant_unfold).numpy().astype("i")
print(input_vu_par_cnn_et_sat_ref)
# print(input_vu_par_cnn_et_sat)
# print(input_vu_par_cnn_et_sat.shape)

a, b, c = input_vu_par_cnn_et_sat_ref.shape

shapeici_out = input_vu_par_cnn_et_sat_ref.shape[-1]

mapping_filter, input_dim = get_mapping_filter(args)

print(mapping_filter)

try:
    feature_pos = quantized_model_train.feature_pos - 1
except:
    feature_pos = 8
W_LR = 1.0 * quantized_model_train.features[feature_pos - 1].weight.detach().cpu().clone().numpy()
# print(model_train.features[feature_pos - 1].bias.data.shape)
b_LR = 1.0 * quantized_model_train.features[feature_pos - 1].bias.detach().cpu().clone().numpy()

# with torch.no_grad():
#     _, _, shape_all_tensorinput_block, shape_all_tensoroutput_block = inference_layer_by_layer_scale(
#         image.to(device), quantized_model_train)
#     if args.first_layer=="bin" or args.first_layer=="float":
#         block_ref_all_inputs, block_ref_all_outputs = get_refs_all(shape_all_tensorinput_block, shape_all_tensoroutput_block)
#     elif args.first_layer == "BNs":
#         coeff = 1
#         if args.dataset == "CIFAR10":
#             coeff = 3
#         # block_ref_all_inputs, block_ref_all_outputs, cptfinal = get_refs_all2(shape_all_tensorinput_block,
#         #                                                                           shape_all_tensoroutput_block, coeff)
#
#
# print()
#
# dictionnary_ref = get_dictionnary_ref(args, block_ref_all_inputs, block_ref_all_outputs, unfold_all,
#                         mapping_filter)
# features1_ref = block_ref_all_outputs[list(block_ref_all_outputs.keys())[-1]].reshape(-1).clone().numpy().astype('i')

if args.fairnessornot:
    txtfin="Fairness_"
else:
    txtfin = ""
_, all_dnf, nogolist = load_cnf_dnf(args, shapeici_out, txtfin = txtfin)


print("(Block, Filter) at False", nogolist)



import time
import random
total = 0
soublock = 1
correct = 0
dataloaders, trainloader, valloader, testloader  = load_data_modt(args)
# with open(os.path.join(DATA_DIR, args.dataset + '_original_mapping.json')) as jfile:
#    mapping = json.loads(jfile)
##permutation =
# original_loader =
feature_start = quantized_model_train.feature_start
sensitive_idx_list = []
for mappingindex in list(mappings.keys()):
    mapping = mappings[mappingindex]
    offset = len(list(mapping.keys()))
    keys = list(mapping.keys())
    sens_values = []
    if type(args.sensitive_fairness) == list:
        sens_values = [mapping[str(i)] for i in args.sensitive_fairness]
        new_idx = [keys.index(str(i)) for i in args.sensitive_fairness]
    else:
        sens_values = mapping[str(args.sensitive_fairness)]
        new_idx = keys.index(str(args.sensitive_fairness))
    #sensitive_idx = new_idx
    #print(new_idx)
    for new_idx2 in new_idx:
        sensitive_idx_list.append(new_idx2 + int(mappingindex) * offset)

    print("Sensitive features : ", sensitive_idx_list)


#P(S>50)
NT_N=0
NT_D=0
N_tot_C1_A1 = 0
N_tot_C0_A1 = 0
N_tot_C1_A0 = 0
N_tot_C0_A0 = 0
with torch.no_grad():
    #for phase in ["train"]:
    running_corrects = 0.0
    nbre_sample = 0
    tk0 = tqdm(dataloaders["train"], total=int(len(dataloaders["train"])))
    for i, data in enumerate(tk0):
        inputs, labels = data
        predicted_labels = np.argmax(labels.cpu().detach().numpy(), axis=-1)
        NT_N+= np.sum(predicted_labels)
        NT_D+= predicted_labels.shape[0]
        #for batchici in range(inputs.shape[0]):
            #if labels[batchici] and inputs[batchici, sensitive_idx]:
            #    N_tot_C1_A1+=1
            #if labels[batchici] and (not inputs[batchici, sensitive_idx]):
            #    N_tot_C1_A0+=1
            #if (not labels[batchici]) and inputs[batchici, sensitive_idx]:
            #    N_tot_C0_A1+=1
            #if (not labels[batchici]) and (not inputs[batchici, sensitive_idx]):
            #    N_tot_C0_A0+=1


print(" P(C=1) = ", NT_N/NT_D)
print(" P(C=0) = ", (NT_D-NT_N)/NT_D)
#print(" P(C=1, A=1) = ", N_tot_C1_A1/NT_D)
#print(" P(C=1, A=0) = ", N_tot_C1_A0/NT_D)
#print(" P(C=0, A=1) = ", N_tot_C0_A1/NT_D)
#print(" P(C=0, A=0) = ", N_tot_C0_A0/NT_D)

#assert NT_D==N_tot_C1_A1+N_tot_C1_A1+N_tot_C1_A1+N_tot_C1_A1
#assert N_tot_C1_A1+ N_tot_C1_A1==NT_N


print(" # (C=1) = ", NT_N)
print(" # (C=0) = ", NT_D-NT_N)

Proba_N_R_L1={}
Proba_N_R_L0={}
Proba_N_R_L1_A_0={}
Proba_N_R_L1_A_1={}
Proba_N_R_L0_A_0={}
Proba_N_R_L0_A_1={}
for block_occurence in range(len(args.Blocks_filters_output)):
    Proba_N_R_L1[block_occurence] = {}
    Proba_N_R_L0[block_occurence] = {}
    Proba_N_R_L1_A_0[block_occurence] = {}
    Proba_N_R_L1_A_1[block_occurence] = {}
    Proba_N_R_L0_A_0[block_occurence] = {}
    Proba_N_R_L0_A_1[block_occurence] = {}
    for filter_occurence in range(args.Blocks_filters_output[block_occurence]):
        Proba_N_R_L1[block_occurence][filter_occurence] = {}
        Proba_N_R_L0[block_occurence][filter_occurence] = {}
        Proba_N_R_L1_A_0[block_occurence][filter_occurence] = {}
        Proba_N_R_L1_A_1[block_occurence][filter_occurence] = {}
        Proba_N_R_L0_A_0[block_occurence][filter_occurence] = {}
        Proba_N_R_L0_A_1[block_occurence][filter_occurence] = {}
        for xy_pixel in range(shapeici_out):
            Proba_N_R_L1[block_occurence][filter_occurence][xy_pixel] = 0
            Proba_N_R_L0[block_occurence][filter_occurence][xy_pixel] = 0
            Proba_N_R_L1_A_0[block_occurence][filter_occurence][xy_pixel] = 0
            Proba_N_R_L1_A_1[block_occurence][filter_occurence][xy_pixel] = 0
            Proba_N_R_L0_A_0[block_occurence][filter_occurence][xy_pixel] = 0
            Proba_N_R_L0_A_1[block_occurence][filter_occurence][xy_pixel] = 0


print()
print(" START RUNNING DDT ")
print()

Pred_all0 = None
Pred_all1 = None
LAbbel_all = None
Pred_allN = None
dp_scores = []
opp_scores = []
with torch.no_grad():
    for phase in ["train"]:
        running_corrects = 0.0
        nbre_sample = 0
        if phase == "train":
            tk0 = tqdm(dataloaders[phase], total=int(len(dataloaders[phase])))
        else:
            tk0 = tqdm(dataloaders["val"], total=int(len(dataloaders["val"])))
        for i, data in enumerate(tk0):
            inputs, labels = data
            inputs0 = deepcopy(inputs)
            inputs1 = deepcopy(inputs)
            for sensitive_idx in sensitive_idx_list:
                # print(inputs0[:, sensitive_idx])
                inputs0[:, sensitive_idx] = 0
                # print(inputs0[:, sensitive_idx])
                # print(inputs1[:, sensitive_idx])
                inputs1[:, sensitive_idx] = 1
            # permutation = np.array(list(mapping.keys())).astype(int)
            # inputs = inputs[:, permutation]
            # print(permutation, inputs.shape)
            batch_size_test = labels.shape[0]
            res_all_tensorinput_block, res_all_tensoroutput_block, shape_all_tensorinput_block, shape_all_tensoroutput_block = inference_layer_by_layer_scale(
                inputs.to(device), quantized_model_train)

            res_all_tensorinput_block0, res_all_tensoroutput_block0, _, _ = inference_layer_by_layer_scale(
                inputs0.to(device), quantized_model_train)

            res_all_tensorinput_block1, res_all_tensoroutput_block1, _, _ = inference_layer_by_layer_scale(
                inputs1.to(device), quantized_model_train)

            nbre_sample += inputs.shape[0]
            predicted_labels = np.argmax(labels.cpu().detach().numpy(), axis=-1)
            for block_occurence in range(len(args.Blocks_filters_output)):
                if len(res_all_tensorinput_block[block_occurence]) == 1:
                    iterici = 0
                else:
                    iterici = soublock
                # print(len(res_all_tensorinput_block[block_occurence]))
                imgs_debut = res_all_tensorinput_block[block_occurence][iterici]  # .unsqueeze(0)
                imgs_debut0 = res_all_tensorinput_block0[block_occurence][iterici]
                imgs_debut1 = res_all_tensorinput_block1[block_occurence][iterici]
                filtericitot = args.Blocks_filters_output[block_occurence]
                imgs_finm1 = res_all_tensoroutput_block[block_occurence][iterici]
                imgs_fin = torch.zeros_like(imgs_finm1)
                imgs_fin0 = torch.zeros_like(imgs_finm1)
                imgs_fin1 = torch.zeros_like(imgs_finm1)
                unfold_block = unfold_all[block_occurence][iterici]
                nombredefiltredansgroupe = int(imgs_debut.shape[1] / int(filtericitot / args.groups_per_block[iterici]))
                outfilter = max(1, int(imgs_fin.shape[1] / imgs_debut.shape[1]))
                shapeici_out2 = imgs_fin.shape[-1]
                for filter_occurence in range(args.Blocks_filters_output[block_occurence]):
                    print_flag_error = True
                    dnf_local = all_dnf[block_occurence][filter_occurence]
                    #cnf_local = all_cnf[block_occurence][filter_occurence]
                    #print(imgs_debut.shape)
                    input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                    mapping_filter[block_occurence][filter_occurence]
                                                    : mapping_filter[block_occurence][filter_occurence] + int(
                                                        args.groups_per_block[block_occurence]),
                                                    :].unsqueeze(-1)
                    input_vu_par_cnn_avant_unfold0 = imgs_debut0[:,
                                                    mapping_filter[block_occurence][filter_occurence]
                                                    : mapping_filter[block_occurence][filter_occurence] + int(
                                                        args.groups_per_block[block_occurence]),
                                                    :].unsqueeze(-1)
                    input_vu_par_cnn_avant_unfold1 = imgs_debut1[:,
                                                    mapping_filter[block_occurence][filter_occurence]
                                                    : mapping_filter[block_occurence][filter_occurence] + int(
                                                        args.groups_per_block[block_occurence]),
                                                    :].unsqueeze(-1)
                    #:]
                    #print(input_vu_par_cnn_avant_unfold.shape)
                    # print(unfold_block)
                    input_vu_par_cnn_et_sat = unfold_block(input_vu_par_cnn_avant_unfold).numpy().astype("i")  #
                    input_vu_par_cnn_et_sat0 = unfold_block(input_vu_par_cnn_avant_unfold0).numpy().astype("i")  #
                    input_vu_par_cnn_et_sat1 = unfold_block(input_vu_par_cnn_avant_unfold1).numpy().astype("i")  #
                    # print(block_occurence, filter_occurence)
                    #print(input_vu_par_cnn_et_sat.shape)
                    for batchici in range(imgs_fin.shape[0]):
                        # output_vu_par_cnn_et_sat = imgs_fin[batchici, filter_occurence, :, :]#.reshape(-1)
                        shapeici_out = input_vu_par_cnn_et_sat.shape[-1]
                        for xy_pixel in range(shapeici_out):


                            if (block_occurence, filter_occurence, xy_pixel) not in nogolist:
                                # print(ok)
                                input_var = input_vu_par_cnn_et_sat[batchici, :, xy_pixel].tolist()
                                input_var0 = input_vu_par_cnn_et_sat0[batchici, :, xy_pixel].tolist()
                                input_var1 = input_vu_par_cnn_et_sat1[batchici, :, xy_pixel].tolist()
                                input_var_int = int("".join(str(x) for x in input_var), 2)
                                input_var_int0 = int("".join(str(x) for x in input_var0), 2)
                                input_var_int1 = int("".join(str(x) for x in input_var1), 2)
                                # print(dnf_local)
                                if type(dnf_local) is int:
                                    pass
                                    # test_output_sat = 0
                                else:
                                    flagDNF = True
                                    for filter_occurence2 in range(filter_occurence+1):
                                        #print((filter_occurence2, filter_occurence, xy_pixel),  args.filters_correlation_pos)
                                        if (filter_occurence2, filter_occurence, xy_pixel) in args.filters_correlation_pos and flagDNF:
                                            test_output_sat_dnf =  int(copy(imgs_fin[batchici, filter_occurence2, xy_pixel]))
                                            test_output_sat_dnf0 = int(copy(imgs_fin0[batchici, filter_occurence2, xy_pixel]))
                                            test_output_sat_dnf1 = int(copy(imgs_fin1[batchici, filter_occurence2, xy_pixel]))
                                            flagDNF= False
                                            #print((filter_occurence2, filter_occurence, xy_pixel))
                                        if (filter_occurence2, filter_occurence, xy_pixel) in args.filters_correlation_neg and flagDNF:
                                            test_output_sat_dnf =  (int(copy(imgs_fin[batchici, filter_occurence2, xy_pixel]))+1)%2
                                            test_output_sat_dnf0 = (int(copy(imgs_fin0[batchici, filter_occurence2, xy_pixel]))+1)%2
                                            test_output_sat_dnf1 = (int(copy(imgs_fin1[batchici, filter_occurence2, xy_pixel]))+1)%2
                                            flagDNF= False
                                            #print((filter_occurence2, filter_occurence, xy_pixel))
                                    if flagDNF:
                                        test_output_sat_dnf = int(dnf_local[xy_pixel][input_var_int])
                                        test_output_sat_dnf0 = int(dnf_local[xy_pixel][input_var_int0])
                                        test_output_sat_dnf1 = int(dnf_local[xy_pixel][input_var_int1])

                                    imgs_fin[batchici, filter_occurence, xy_pixel] = test_output_sat_dnf
                                    imgs_fin0[batchici, filter_occurence, xy_pixel] = test_output_sat_dnf0
                                    imgs_fin1[batchici, filter_occurence, xy_pixel] = test_output_sat_dnf1

                                    if predicted_labels[batchici] and test_output_sat_dnf:
                                        Proba_N_R_L1[block_occurence][filter_occurence][xy_pixel] += 1
                                        if inputs[batchici, sensitive_idx].item() :
                                            Proba_N_R_L1_A_1[block_occurence][filter_occurence][xy_pixel] += 1
                                        else:
                                            Proba_N_R_L1_A_0[block_occurence][filter_occurence][xy_pixel] += 1

                                    elif (not predicted_labels[batchici]) and test_output_sat_dnf:
                                        Proba_N_R_L0[block_occurence][filter_occurence][xy_pixel] += 1
                                        #print(sensitive_idx)
                                        #print(batchici, inputs[batchici, sensitive_idx])
                                        if inputs[batchici, sensitive_idx].item() :
                                            Proba_N_R_L0_A_1[block_occurence][filter_occurence][xy_pixel] += 1
                                        else:
                                            Proba_N_R_L0_A_0[block_occurence][filter_occurence][xy_pixel] += 1





                if block_occurence + 1 < (len(args.Blocks_filters_output)):
                    res_all_tensorinput_block[block_occurence + 1][iterici] = copy(imgs_fin)
                    res_all_tensorinput_block0[block_occurence + 1][iterici] = copy(imgs_fin0)
                    res_all_tensorinput_block1[block_occurence + 1][iterici] = copy(imgs_fin1)
                else:
                    features_replace = imgs_fin.view(batch_size_test, -1).numpy().astype('i')
                    features_replace0 = imgs_fin0.view(batch_size_test, -1).numpy().astype('i')
                    features_replace1 = imgs_fin1.view(batch_size_test, -1).numpy().astype('i')
                # print(block_occurence, torch.sum(imgs_fin == -1) / batch_size_test)
            feature_vector = imgs_fin.view(batch_size_test, -1).numpy().astype(
                'i').transpose()
            feature_vector0 = imgs_fin0.view(batch_size_test, -1).numpy().astype(
                'i').transpose()
            feature_vector1 = imgs_fin1.view(batch_size_test, -1).numpy().astype(
                'i').transpose()
            V_ref = np.dot(W_LR, feature_vector).transpose() + b_LR
            V_ref0 = np.dot(W_LR, feature_vector0).transpose() + b_LR
            V_ref1 = np.dot(W_LR, feature_vector1).transpose() + b_LR
            predicted = np.argmax(V_ref, axis=-1)
            predicted0 = np.argmax(V_ref0, axis=-1)
            predicted1 = np.argmax(V_ref1, axis=-1)
            predicted_labels = np.argmax(labels.cpu().detach().numpy(), axis=-1)
            # print(predicted)
            running_corrects += (predicted == predicted_labels).sum().item()
            acc = running_corrects / nbre_sample

            if Pred_all0 is None:
                Pred_all0 = predicted0
                Pred_all1 = predicted1
                LAbbel_all = predicted_labels
                Pred_allN = predicted
            else:
                Pred_allN = np.concatenate((predicted, Pred_allN), axis=0)
                Pred_all0 = np.concatenate((predicted0, Pred_all0), axis=0)
                Pred_all1 = np.concatenate((predicted0, Pred_all1), axis=0)
                LAbbel_all = np.concatenate((predicted_labels, LAbbel_all), axis=0)
        print('{} Acc: {:.4f}'.format(
            phase, acc))
        dp_scores.append(dp(Pred_all0, Pred_all1))
        # print(dp_scores)
        opp_scores.append(opportunity(Pred_all0, Pred_all1, LAbbel_all))
        print("dp_scores", dp_scores[0])
        print("opp_scores", opp_scores[0])
        # print(ok)
        print('Precision: %.3f' % precision_score(Pred_allN, LAbbel_all))
        print('Recall: %.3f' % recall_score(Pred_allN, LAbbel_all))
        print('F1 Score: %.3f' % f1_score(Pred_allN, LAbbel_all))
        conf_matrixres = confusion_matrix(y_true=Pred_allN, y_pred=LAbbel_all)
        print(conf_matrixres)

P_S_F=0
P_F=0
REGLE_PROBA = {}
P_RI = {}
cpt_rule = 0
pos_neg = None
if args.dc_logic:
    if args.fairnessornot:
        base = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(
            " ", "") + '/avec_DC_logic/Filtrage_Fairness' + str(args.filtrage) + "/"
    else:
        base = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(
            " ", "") + '/avec_DC_logic/Filtrage_' + str(args.filtrage) + "/"
else:
    if args.fairnessornot:
        base = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(
            " ", "") + '/sans_DC_logic/Filtrage_Fairness' + str(args.filtrage) + "/"
    else:
        base = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(
            " ", "") + '/sans_DC_logic/Filtrage_' + str(args.filtrage) + "/"


liste_keep_fair = []
for block_occurence in range(len(args.Blocks_filters_output)):
    REGLE_PROBA[block_occurence]={}
    P_RI[block_occurence]={}
    for filter_occurence in range(args.Blocks_filters_output[block_occurence]):
        REGLE_PROBA[block_occurence][filter_occurence]={}
        P_RI[block_occurence][filter_occurence]={}
        for xy_pixel in range(shapeici_out):
            if Proba_N_R_L1[block_occurence][filter_occurence][xy_pixel] != 0 or Proba_N_R_L0[block_occurence][filter_occurence][xy_pixel] != 0:
                #print(Proba_N_R_L1[block_occurence][filter_occurence][xy_pixel], Proba_N_R_L0[block_occurence][filter_occurence][xy_pixel])
                cpt_rule +=1
                print()
                print(args.HK[3*cpt_rule-3])




                if W_LR[1][filter_occurence * shapeici_out + xy_pixel].item() != 0:
                    pos_neg = "pos"
                    P_r1_s = Proba_N_R_L1[block_occurence][filter_occurence][xy_pixel]/NT_N
                    P_r1_nots = Proba_N_R_L0[block_occurence][filter_occurence][xy_pixel]/(NT_D-NT_N)
                    NUM = P_r1_s * NT_N/NT_D
                    DENUM = NUM + (1-(NT_N/NT_D))*P_r1_nots

                    print(" % C=1 | REGLE ACTIVE ", 100*NUM/DENUM)
                    print(" % REGLE ACTIVE ", 100*DENUM)
                    REGLE_PROBA[block_occurence][filter_occurence][xy_pixel] = NUM/DENUM
                    P_RI[block_occurence][filter_occurence][xy_pixel] = DENUM

                    P_r1_s = Proba_N_R_L1_A_1[block_occurence][filter_occurence][xy_pixel] / NT_N
                    P_r1_nots = Proba_N_R_L0_A_1[block_occurence][filter_occurence][xy_pixel] / (NT_D - NT_N)
                    NUM_L1_A1 = P_r1_s * NT_N/NT_D
                    DENUM_L1_A1 = NUM_L1_A1 + (1 - (NT_N / NT_D)) * P_r1_nots

                    print(" % C=1 | REGLE ACTIVE, A= 1 ", 100 * NUM_L1_A1 / DENUM_L1_A1)
                    print(" % REGLE ACTIVE, A=1 ", 100 * DENUM_L1_A1)

                    P_r1_s = Proba_N_R_L1_A_0[block_occurence][filter_occurence][xy_pixel] / NT_N
                    P_r1_nots = Proba_N_R_L0_A_0[block_occurence][filter_occurence][xy_pixel] / (NT_D - NT_N)
                    NUM10 = P_r1_s * NT_N / NT_D
                    DENUM10 = NUM10 + (1 - (NT_N / NT_D)) * P_r1_nots

                    print(" % C=1 | REGLE ACTIVE, A= 0 ", 100 * NUM10 / DENUM10)
                    print(" % REGLE ACTIVE, A=0 ", 100 * DENUM10)




                if W_LR[0][filter_occurence * shapeici_out + xy_pixel].item() != 0:
                    pos_neg = "neg"
                    P_r1_s = Proba_N_R_L1[block_occurence][filter_occurence][xy_pixel]/NT_N
                    P_r1_nots = Proba_N_R_L0[block_occurence][filter_occurence][xy_pixel]/(NT_D-NT_N)
                    NUM = P_r1_nots * (1-(NT_N/NT_D))
                    DENUM = NUM + P_r1_s * NT_N/NT_D

                    print(" % C=0 | REGLE ACTIVE ", 100*NUM/DENUM)
                    print(" % REGLE ACTIVE ", 100*DENUM)
                    REGLE_PROBA[block_occurence][filter_occurence][xy_pixel] = -1*NUM/DENUM
                    P_RI[block_occurence][filter_occurence][xy_pixel] = DENUM

                    P_r1_s = Proba_N_R_L1_A_1[block_occurence][filter_occurence][xy_pixel] / NT_N
                    P_r1_nots = Proba_N_R_L0_A_1[block_occurence][filter_occurence][xy_pixel] / (NT_D - NT_N)
                    NUM_L1_A1 = P_r1_nots * (1 - (NT_N / NT_D))
                    DENUM_L1_A1 =NUM_L1_A1 + NT_N/NT_D * P_r1_nots

                    print(" % C=0 | REGLE ACTIVE, A= 1 ", 100 * NUM_L1_A1 / DENUM_L1_A1)
                    print(" % REGLE ACTIVE, A=1 ", 100 * DENUM_L1_A1)

                    P_r1_s = Proba_N_R_L1_A_0[block_occurence][filter_occurence][xy_pixel] / NT_N
                    P_r1_nots = Proba_N_R_L0_A_0[block_occurence][filter_occurence][xy_pixel] / (NT_D - NT_N)
                    NUM10 = P_r1_nots * (1 - (NT_N / NT_D))
                    DENUM10 = NUM10 + NT_N/NT_D * P_r1_nots

                    print(" % C=0| REGLE ACTIVE, A= 0 ", 100 * NUM10 / DENUM10)
                    print(" % REGLE ACTIVE, A=0 ", 100 * DENUM10)


                with open(base + "/" + str(args.load_general_exception_filters) + "_RULE_"+str(cpt_rule)+"_"+str(filter_occurence)+"_"+str(xy_pixel)+"_"+str(pos_neg)+"_probas6F_", 'w') as f:
                    f.write(str([NUM/DENUM, DENUM, NUM_L1_A1 / DENUM_L1_A1, DENUM_L1_A1, NUM10 / DENUM10,  DENUM10]))

                if args.dataset in args.dataset in ["adult", "law", "diabetes"]:
                    thr_corr = 0.05
                else:
                    thr_corr = 0.1
                if abs(NUM_L1_A1 / DENUM_L1_A1 -  NUM10 / DENUM10) < thr_corr : #and filteroccurence != filteroccurence2:
                    #if (filteroccurence2, filteroccurence, xy_pixel) not in liste_correlation_positive:
                    liste_keep_fair.append((block_occurence, filter_occurence, xy_pixel))
                    print("OK")
                #if corr_vf < -1 * thr_corr and filteroccurence != filteroccurence2:
                #    if (filteroccurence2, filteroccurence, xy_pixel) not in liste_correlation_negative:
                #        liste_correlation_negative.append((filteroccurence, filteroccurence2, xy_pixel))
print(liste_keep_fair)
if args.load_general:
    if args.dc_logic:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/avec_DC_logic/Filtrage_' + str(
            txtfin) + str(
            args.filtrage) + "/"
        # path_save_modelvf2 = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
        #                                                                                            "") + '/avec_DC_logic/Filtrage_' + str(0)  + "/"
    else:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/sans_DC_logic/Filtrage_' + str(
            txtfin) + str(
            args.filtrage) + "/"
        # path_save_modelvf2 = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
        #                                                                                             "") + '/sans_DC_logic/Filtrage_' + str(
        #    0) + "/"
else:
    path_save_modelvf = args.load_specific
with open(path_save_modelvf + "liste_keep_fair.txt", 'w') as f:
    f.write(str(liste_keep_fair))
"""
print()
print(" START FINDING THR OPTI ")
print()
predicted_proba_all=[]
#predicted_proba_all0=[]
#predicted_proba_all1=[]

label_all = []
with torch.no_grad():
    for phase in ["val"]:
        running_corrects = 0.0
        nbre_sample = 0
        #if phase == "train":
        tk0 = tqdm(dataloaders[phase], total=int(len(dataloaders[phase])))
        #else:
        #    tk0 = tqdm(dataloaders["val"], total=int(len(dataloaders["val"])))
        for i, data in enumerate(tk0):
            inputs, labels = data
            predicted_labels = np.argmax(labels.cpu().detach().numpy(), axis=-1)
            #inputs0 = deepcopy(inputs)
            #inputs1 = deepcopy(inputs)
            batch_size_test = labels.shape[0]
            proba_test = [P_S_F * P_F] * batch_size_test
            #proba_test0 = [P_S_F * P_F] * batch_size_test
            #proba_test1 = [P_S_F * P_F] * batch_size_test
            #for sensitive_idx in sensitive_idx_list:
                # print(inputs0[:, sensitive_idx])
            #    inputs0[:, sensitive_idx] = 0
                # print(inputs0[:, sensitive_idx])
                # print(inputs1[:, sensitive_idx])
            #    inputs1[:, sensitive_idx] = 1
            # permutation = np.array(list(mapping.keys())).astype(int)
            # inputs = inputs[:, permutation]
            # print(permutation, inputs.shape)
            batch_size_test = labels.shape[0]
            res_all_tensorinput_block, res_all_tensoroutput_block, shape_all_tensorinput_block, shape_all_tensoroutput_block = inference_layer_by_layer_scale(
                inputs.to(device), quantized_model_train)

            #res_all_tensorinput_block0, res_all_tensoroutput_block0, _, _ = inference_layer_by_layer_scale(
            #    inputs0.to(device), quantized_model_train)

            #res_all_tensorinput_block1, res_all_tensoroutput_block1, _, _ = inference_layer_by_layer_scale(
            #    inputs1.to(device), quantized_model_train)

            nbre_sample += inputs.shape[0]
            for block_occurence in range(len(args.Blocks_filters_output)):
                if len(res_all_tensorinput_block[block_occurence]) == 1:
                    iterici = 0
                else:
                    iterici = soublock
                # print(len(res_all_tensorinput_block[block_occurence]))
                imgs_debut = res_all_tensorinput_block[block_occurence][iterici]  # .unsqueeze(0)
                #imgs_debut0 = res_all_tensorinput_block0[block_occurence][iterici]
                #imgs_debut1 = res_all_tensorinput_block1[block_occurence][iterici]
                filtericitot = args.Blocks_filters_output[block_occurence]
                imgs_fin = res_all_tensoroutput_block[block_occurence][iterici]
                #imgs_fin0 = res_all_tensoroutput_block0[block_occurence][iterici]
                #imgs_fin1 = res_all_tensoroutput_block1[block_occurence][iterici]
                unfold_block = unfold_all[block_occurence][iterici]
                nombredefiltredansgroupe = int(imgs_debut.shape[1] / int(filtericitot / args.groups_per_block[iterici]))
                outfilter = max(1, int(imgs_fin.shape[1] / imgs_debut.shape[1]))
                shapeici_out2 = imgs_fin.shape[-1]
                for filter_occurence in range(args.Blocks_filters_output[block_occurence]):
                    print_flag_error = True
                    dnf_local = all_dnf[block_occurence][filter_occurence]
                    #cnf_local = all_cnf[block_occurence][filter_occurence]
                    # print(imgs_debut.shape)
                    input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                    mapping_filter[block_occurence][filter_occurence]
                                                    : mapping_filter[block_occurence][filter_occurence] + int(
                                                        args.groups_per_block[block_occurence]),
                                                    :].unsqueeze(-1)
                    #input_vu_par_cnn_avant_unfold0 = imgs_debut0[:,
                    #                                 mapping_filter[block_occurence][filter_occurence]
                    #                                 : mapping_filter[block_occurence][filter_occurence] + int(
                    #                                     args.groups_per_block[block_occurence]),
                    #                                 :].unsqueeze(-1)
                    #input_vu_par_cnn_avant_unfold1 = imgs_debut1[:,
                    #                                 mapping_filter[block_occurence][filter_occurence]
                    #                                 : mapping_filter[block_occurence][filter_occurence] + int(
                    #                                     args.groups_per_block[block_occurence]),
                    #                                 :].unsqueeze(-1)
                    #:]
                    # print(input_vu_par_cnn_avant_unfold.shape)
                    # print(unfold_block)
                    input_vu_par_cnn_et_sat = unfold_block(input_vu_par_cnn_avant_unfold).numpy().astype("i")  #
                    #input_vu_par_cnn_et_sat0 = unfold_block(input_vu_par_cnn_avant_unfold0).numpy().astype("i")  #
                    #input_vu_par_cnn_et_sat1 = unfold_block(input_vu_par_cnn_avant_unfold1).numpy().astype("i")  #
                    # print(block_occurence, filter_occurence)
                    # print(input_vu_par_cnn_et_sat.shape)
                    for batchici in range(imgs_fin.shape[0]):
                        # output_vu_par_cnn_et_sat = imgs_fin[batchici, filter_occurence, :, :]#.reshape(-1)
                        shapeici_out = input_vu_par_cnn_et_sat.shape[-1]
                        for xy_pixel in range(shapeici_out):

                            if (block_occurence, filter_occurence, xy_pixel) not in nogolist:
                                # print(ok)
                                input_var = input_vu_par_cnn_et_sat[batchici, :, xy_pixel].tolist()
                                #input_var0 = input_vu_par_cnn_et_sat0[batchici, :, xy_pixel].tolist()
                                #input_var1 = input_vu_par_cnn_et_sat1[batchici, :, xy_pixel].tolist()
                                input_var_int = int("".join(str(x) for x in input_var), 2)
                                #input_var_int0 = int("".join(str(x) for x in input_var0), 2)
                                #input_var_int1 = int("".join(str(x) for x in input_var1), 2)
                                # print(dnf_local)
                                if type(dnf_local) is int:
                                    pass
                                    # test_output_sat = 0
                                else:
                                    flagDNF = True
                                    for filter_occurence2 in range(filter_occurence + 1):
                                        # print((filter_occurence2, filter_occurence, xy_pixel),  args.filters_correlation_pos)
                                        if (filter_occurence2, filter_occurence,
                                            xy_pixel) in args.filters_correlation_pos and flagDNF:
                                            test_output_sat_dnf = int(
                                                copy(imgs_fin[batchici, filter_occurence2, xy_pixel]))
                                            #test_output_sat_dnf0 = int(
                                            #    copy(imgs_fin0[batchici, filter_occurence2, xy_pixel]))
                                            #test_output_sat_dnf1 = int(
                                            #    copy(imgs_fin1[batchici, filter_occurence2, xy_pixel]))
                                            flagDNF = False
                                            # print((filter_occurence2, filter_occurence, xy_pixel))
                                        if (filter_occurence2, filter_occurence,
                                            xy_pixel) in args.filters_correlation_neg and flagDNF:
                                            test_output_sat_dnf = (int(copy(
                                                imgs_fin[batchici, filter_occurence2, xy_pixel])) + 1) % 2
                                            #test_output_sat_dnf0 = (int(copy(
                                            #    imgs_fin0[batchici, filter_occurence2, xy_pixel])) + 1) % 2
                                            #test_output_sat_dnf1 = (int(copy(
                                            #    imgs_fin1[batchici, filter_occurence2, xy_pixel])) + 1) % 2
                                            flagDNF = False
                                            # print((filter_occurence2, filter_occurence, xy_pixel))
                                    if flagDNF:
                                        test_output_sat_dnf = int(dnf_local[xy_pixel][input_var_int])
                                        test_output_sat_dnf0 = int(dnf_local[xy_pixel][input_var_int0])
                                        test_output_sat_dnf1 = int(dnf_local[xy_pixel][input_var_int1])



                                    if test_output_sat_dnf: #and  W_LR[1][filter_occurence * shapeici_out + xy_pixel].item() != 0:
                                        proba_test[batchici] += P_RI[block_occurence][filter_occurence][xy_pixel] * \
                                                                REGLE_PROBA[block_occurence][filter_occurence][xy_pixel]
                                    #if test_output_sat_dnf0:
                                    #    proba_test0[batchici] += P_RI[block_occurence][filter_occurence][xy_pixel] * \
                                    #                            REGLE_PROBA[block_occurence][filter_occurence][xy_pixel]
                                    #if test_output_sat_dnf1:
                                    #    proba_test1[batchici] += P_RI[block_occurence][filter_occurence][xy_pixel] * \
                                    #                            REGLE_PROBA[block_occurence][filter_occurence][xy_pixel]

            predicted_proba_all += proba_test
            #predicted_proba_all0 += proba_test0
            #predicted_proba_all1 += proba_test1
            label_all += predicted_labels.tolist()


acc_probabest = 0.0
predicted_proba_all = 1.0*np.array(predicted_proba_all)
label_all = 1.0*np.array(label_all)
#print(predicted_proba_all, label_all)
for i in range(200):
    predicted_proba = 1.0 * (predicted_proba_all > i/100)
    #print(predicted_proba, type(predicted_proba))
    #print(label_all, type(label_all))
    #print(predicted_proba.shape, label_all.shape)
    running_corrects_proba = (predicted_proba == label_all).sum().item()
    #print(running_corrects_proba)
    acc_proba = running_corrects_proba / len(label_all)
    #print(i, acc_proba)
    if acc_proba>acc_probabest:
        acc_probabest=acc_proba
        thr = i/100
        #ok

print()
print('{} thr and Acc on VAL PROBA: {:.4f}'.format(
        thr, acc_probabest))


print()
print(" START FINAL ")
print()
Pred_all0 = None
Pred_all1 = None
LAbbel_all = None
Pred_allN = None
dp_scores = []
opp_scores = []

label_all = []
with torch.no_grad():
    for phase in ["test"]:
        running_corrects = 0.0
        nbre_sample = 0
        #if phase == "test":
        tk0 = tqdm(dataloaders[phase], total=int(len(dataloaders[phase])))
        for i, data in enumerate(tk0):
            inputs, labels = data
            inputs0 = deepcopy(inputs)
            inputs1 = deepcopy(inputs)
            batch_size_test = labels.shape[0]
            proba_test = [P_S_F * P_F] * batch_size_test
            proba_test0 = [P_S_F * P_F] * batch_size_test
            proba_test1 = [P_S_F * P_F] * batch_size_test
            for sensitive_idx in sensitive_idx_list:
                # print(inputs0[:, sensitive_idx])
                inputs0[:, sensitive_idx] = 0
                # print(inputs0[:, sensitive_idx])
                # print(inputs1[:, sensitive_idx])
                inputs1[:, sensitive_idx] = 1
            # permutation = np.array(list(mapping.keys())).astype(int)
            # inputs = inputs[:, permutation]
            # print(permutation, inputs.shape)
            batch_size_test = labels.shape[0]
            res_all_tensorinput_block, res_all_tensoroutput_block, shape_all_tensorinput_block, shape_all_tensoroutput_block = inference_layer_by_layer_scale(
                inputs.to(device), quantized_model_train)

            res_all_tensorinput_block0, res_all_tensoroutput_block0, _, _ = inference_layer_by_layer_scale(
                inputs0.to(device), quantized_model_train)

            res_all_tensorinput_block1, res_all_tensoroutput_block1, _, _ = inference_layer_by_layer_scale(
                inputs1.to(device), quantized_model_train)

            nbre_sample += inputs.shape[0]
            for block_occurence in range(len(args.Blocks_filters_output)):
                if len(res_all_tensorinput_block[block_occurence]) == 1:
                    iterici = 0
                else:
                    iterici = soublock
                # print(len(res_all_tensorinput_block[block_occurence]))
                imgs_debut = res_all_tensorinput_block[block_occurence][iterici]  # .unsqueeze(0)
                imgs_debut0 = res_all_tensorinput_block0[block_occurence][iterici]
                imgs_debut1 = res_all_tensorinput_block1[block_occurence][iterici]
                filtericitot = args.Blocks_filters_output[block_occurence]
                imgs_fin = res_all_tensoroutput_block[block_occurence][iterici]
                imgs_fin0 = res_all_tensoroutput_block0[block_occurence][iterici]
                imgs_fin1 = res_all_tensoroutput_block1[block_occurence][iterici]
                unfold_block = unfold_all[block_occurence][iterici]
                nombredefiltredansgroupe = int(imgs_debut.shape[1] / int(filtericitot / args.groups_per_block[iterici]))
                outfilter = max(1, int(imgs_fin.shape[1] / imgs_debut.shape[1]))
                shapeici_out2 = imgs_fin.shape[-1]
                for filter_occurence in range(args.Blocks_filters_output[block_occurence]):
                    print_flag_error = True
                    dnf_local = all_dnf[block_occurence][filter_occurence]
                    #cnf_local = all_cnf[block_occurence][filter_occurence]
                    # print(imgs_debut.shape)
                    input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                    mapping_filter[block_occurence][filter_occurence]
                                                    : mapping_filter[block_occurence][filter_occurence] + int(
                                                        args.groups_per_block[block_occurence]),
                                                    :].unsqueeze(-1)
                    input_vu_par_cnn_avant_unfold0 = imgs_debut0[:,
                                                     mapping_filter[block_occurence][filter_occurence]
                                                     : mapping_filter[block_occurence][filter_occurence] + int(
                                                         args.groups_per_block[block_occurence]),
                                                     :].unsqueeze(-1)
                    input_vu_par_cnn_avant_unfold1 = imgs_debut1[:,
                                                     mapping_filter[block_occurence][filter_occurence]
                                                     : mapping_filter[block_occurence][filter_occurence] + int(
                                                         args.groups_per_block[block_occurence]),
                                                     :].unsqueeze(-1)
                    #:]
                    # print(input_vu_par_cnn_avant_unfold.shape)
                    # print(unfold_block)
                    input_vu_par_cnn_et_sat = unfold_block(input_vu_par_cnn_avant_unfold).numpy().astype("i")  #
                    input_vu_par_cnn_et_sat0 = unfold_block(input_vu_par_cnn_avant_unfold0).numpy().astype("i")  #
                    input_vu_par_cnn_et_sat1 = unfold_block(input_vu_par_cnn_avant_unfold1).numpy().astype("i")  #
                    # print(block_occurence, filter_occurence)
                    # print(input_vu_par_cnn_et_sat.shape)
                    for batchici in range(imgs_fin.shape[0]):
                        # output_vu_par_cnn_et_sat = imgs_fin[batchici, filter_occurence, :, :]#.reshape(-1)
                        shapeici_out = input_vu_par_cnn_et_sat.shape[-1]
                        for xy_pixel in range(shapeici_out):

                            if (block_occurence, filter_occurence, xy_pixel) not in nogolist:
                                # print(ok)
                                input_var = input_vu_par_cnn_et_sat[batchici, :, xy_pixel].tolist()
                                input_var0 = input_vu_par_cnn_et_sat0[batchici, :, xy_pixel].tolist()
                                input_var1 = input_vu_par_cnn_et_sat1[batchici, :, xy_pixel].tolist()
                                input_var_int = int("".join(str(x) for x in input_var), 2)
                                input_var_int0 = int("".join(str(x) for x in input_var0), 2)
                                input_var_int1 = int("".join(str(x) for x in input_var1), 2)
                                # print(dnf_local)
                                if type(dnf_local) is int:
                                    pass
                                    # test_output_sat = 0
                                else:
                                    flagDNF = True
                                    for filter_occurence2 in range(filter_occurence + 1):
                                        # print((filter_occurence2, filter_occurence, xy_pixel),  args.filters_correlation_pos)
                                        if (filter_occurence2, filter_occurence,
                                            xy_pixel) in args.filters_correlation_pos and flagDNF:
                                            test_output_sat_dnf = int(
                                                copy(imgs_fin[batchici, filter_occurence2, xy_pixel]))
                                            test_output_sat_dnf0 = int(
                                                copy(imgs_fin0[batchici, filter_occurence2, xy_pixel]))
                                            test_output_sat_dnf1 = int(
                                                copy(imgs_fin1[batchici, filter_occurence2, xy_pixel]))
                                            flagDNF = False
                                            # print((filter_occurence2, filter_occurence, xy_pixel))
                                        if (filter_occurence2, filter_occurence,
                                            xy_pixel) in args.filters_correlation_neg and flagDNF:
                                            test_output_sat_dnf = (int(copy(
                                                imgs_fin[batchici, filter_occurence2, xy_pixel])) + 1) % 2
                                            test_output_sat_dnf0 = (int(copy(
                                                imgs_fin0[batchici, filter_occurence2, xy_pixel])) + 1) % 2
                                            test_output_sat_dnf1 = (int(copy(
                                                imgs_fin1[batchici, filter_occurence2, xy_pixel])) + 1) % 2
                                            flagDNF = False
                                            # print((filter_occurence2, filter_occurence, xy_pixel))
                                    if flagDNF:
                                        test_output_sat_dnf = int(dnf_local[xy_pixel][input_var_int])
                                        test_output_sat_dnf0 = int(dnf_local[xy_pixel][input_var_int0])
                                        test_output_sat_dnf1 = int(dnf_local[xy_pixel][input_var_int1])



                                    if test_output_sat_dnf:
                                        proba_test[batchici] += P_RI[block_occurence][filter_occurence][xy_pixel] * \
                                                                REGLE_PROBA[block_occurence][filter_occurence][xy_pixel]
                                    if test_output_sat_dnf0:
                                        proba_test0[batchici] += P_RI[block_occurence][filter_occurence][xy_pixel] * \
                                                                REGLE_PROBA[block_occurence][filter_occurence][xy_pixel]
                                    if test_output_sat_dnf1:
                                        proba_test1[batchici] += P_RI[block_occurence][filter_occurence][xy_pixel] * \
                                                                REGLE_PROBA[block_occurence][filter_occurence][xy_pixel]


            label_all += predicted_labels.tolist()
            predicted_proba = 1.0 * (np.array(proba_test) > thr)
            predicted_proba0 = 1.0 * (np.array(proba_test0) > thr)
            predicted_proba1 = 1.0 * (np.array(proba_test1) > thr)
            predicted_labels = np.argmax(labels.cpu().detach().numpy(), axis=-1)
            # print(predicted)
            running_corrects += (predicted_proba == predicted_labels).sum().item()
            acc = running_corrects / nbre_sample

            if Pred_all0 is None:
                Pred_all0 = predicted_proba0
                Pred_all1 = predicted_proba1
                LAbbel_all = predicted_labels
                Pred_allN = predicted_proba
            else:
                Pred_allN = np.concatenate((predicted_proba, Pred_allN), axis=0)
                Pred_all0 = np.concatenate((predicted_proba0, Pred_all0), axis=0)
                Pred_all1 = np.concatenate((predicted_proba1, Pred_all1), axis=0)
                LAbbel_all = np.concatenate((predicted_labels, LAbbel_all), axis=0)

    print('{} Acc: {:.4f}'.format(
        phase, acc))
    dp_scores.append(dp(Pred_all0, Pred_all1))
    # print(dp_scores)
    opp_scores.append(opportunity(Pred_all0, Pred_all1, LAbbel_all))
    print("dp_scores", dp_scores[0])
    print("opp_scores", opp_scores[0])
    # print(ok)
    print('Precision: %.3f' % precision_score(Pred_allN, LAbbel_all))
    print('Recall: %.3f' % recall_score(Pred_allN, LAbbel_all))
    print('F1 Score: %.3f' % f1_score(Pred_allN, LAbbel_all))
    conf_matrixres = confusion_matrix(y_true=Pred_allN, y_pred=LAbbel_all)
    print(conf_matrixres)

    args.json_res_all["accuracy"] = acc
    args.json_res_all["DP"] = dp_scores[0]
    args.json_res_all["OP"] = opp_scores[0]
    args.json_res_all["f1"] = f1_score(Pred_allN, LAbbel_all)
    args.json_res_all["precision"] = precision_score(Pred_allN, LAbbel_all)
    args.json_res_all["recall"] = recall_score(Pred_allN, LAbbel_all)
    args.json_res_all["confusion_matrix"] = conf_matrixres.tolist()
"""
