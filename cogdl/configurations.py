from cogdl.experiments import check_experiment
from tabulate import tabulate
import json
import os

def load_hyperparameter_config():
    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'configs.json'
    with open(path, 'r') as file:
        configuration = json.load(file)
    return configuration

def save_hyperparameter_config(configuration):
    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'configs.json'
    with open(path, 'w') as file:
        json.dump(configuration, file, indent=4, ensure_ascii=False)
    return configuration

def result_table(dataset=None, model=None, print_test_accuracy=True):
    configuration = load_hyperparameter_config()
    model = sorted(list(set([_mk for _mk, _mv in configuration.items() for _dk, _dv  in _mv.items() if _dk!='general' and 'val_acc_mean' in _dv]))) if model is None else model
    dataset = sorted(list(set([_dk for _mk, _mv in configuration.items() for _dk, _dv  in _mv.items() if _dk!='general' and 'val_acc_mean' in _dv]))) if dataset is None else dataset

    col_names = ["Variant"] + dataset
    tab_data = []
    for m in model:
        results = [m]
        for d in dataset:
            items = configuration.get(m, {}).get(d, {})
            if 'val_acc_mean' in items: 
                if print_test_accuracy:
                    acc_str = "%5.2f±%4.2f / %5.2f±%4.2f" % (items['test_acc_mean']*100, items['test_acc_std']*100, items['val_acc_mean']*100, items['val_acc_std']*100)
                else:
                    acc_str = "%5.2f±%4.2f" % (items['test_acc_mean']*100, items['test_acc_std']*100)
            else:
                acc_str = ''
            results.append(acc_str)
        tab_data.append(results)
    
    print(tabulate(tab_data, headers=col_names, tablefmt="github", stralign=u'center'))

def result_check(dataset=None, model=None):
    configuration = load_hyperparameter_config()
    model = sorted(list(set([_mk for _mk, _mv in configuration.items() for _dk, _dv  in _mv.items() if _dk!='general' and 'val_acc_mean' in _dv]))) if model is None else model
    dataset = sorted(list(set([_dk for _mk, _mv in configuration.items() for _dk, _dv  in _mv.items() if _dk!='general' and 'val_acc_mean' in _dv]))) if dataset is None else dataset
    
    col_names = ["Variant"] + ["Record", "Experiment", "Deviation"]
    tab_data = []
    
    for d in dataset:
        for m in model:
            items = configuration.get(m, {}).get(d, {})
            if 'val_acc_mean' in items: 
                record_val_acc_mean = items['val_acc_mean']
                record_val_acc_std = items['val_acc_std']
                record_test_acc_mean = items["test_acc_mean"]
                record_test_acc_std = items['test_acc_std']

                result_mean = check_experiment(dataset=d, model=m, use_best_config=True)
                
                experiment_val_acc_mean = result_mean['val_acc_mean']
                experiment_val_acc_std = result_mean['val_acc_std']
                experiment_test_acc_mean = result_mean["test_acc_mean"]
                experiment_test_acc_std = result_mean['test_acc_std']

                deviation_valid_acc_mean = abs(experiment_val_acc_mean - record_val_acc_mean)
                deviation_valid_acc_std = abs(experiment_val_acc_std - record_val_acc_std)
                deviation_test_acc_mean = abs(experiment_test_acc_mean - record_test_acc_mean)
                deviation_test_acc_std = abs(experiment_test_acc_std - record_test_acc_std)

                record_acc = "%5.2f±%4.2f / %5.2f±%4.2f" % (record_test_acc_mean*100, record_test_acc_std*100, record_val_acc_mean*100, record_val_acc_std*100)
                experiment_acc = "%5.2f±%4.2f / %5.2f±%4.2f" % (experiment_test_acc_mean*100, experiment_test_acc_std*100, experiment_val_acc_mean*100, experiment_val_acc_std*100)
                deviation_acc = "%4.2f±%4.2f / %4.2f±%4.2f" % (deviation_test_acc_mean*100, deviation_test_acc_std*100, deviation_valid_acc_mean*100, deviation_valid_acc_std*100)

                tab_data.append(["%s, %s" % (d, m), record_acc, experiment_acc, deviation_acc])
    
    print(tabulate(tab_data, headers=col_names, tablefmt="github", stralign=u'center'))
