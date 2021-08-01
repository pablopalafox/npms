import argparse


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_non_negative(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid non-negative int value" % value)
    return ivalue

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_dataset_type_from_dataset_name(dataset_name):
    if "MANO" in dataset_name.upper():
        return "MANO"
    
    if "AMASS" in dataset_name.upper():
        return "HUMAN"

    if "MIXAMO" in dataset_name.upper():
        return "HUMAN"

    if "CAPE" in dataset_name.upper():
        return "HUMAN"

    if "SMAL" in dataset_name.upper():
        return "SMAL"

    if "DFAUST" in dataset_name.upper():
        return "HUMAN"

    print(dataset_name.upper())

    raise Exception(f"Invalid dataset '{dataset_name}'")

def get_dataset_class_from_dataset_name(dataset_name):
    if "MANO" in dataset_name.upper():
        return "MANO"
    
    if "AMASS" in dataset_name.upper():
        return "AMASS"

    if "MIXAMO" in dataset_name.upper():
        return "MIXAMO"

    if "CAPE" in dataset_name.upper():
        return "CAPE"

    if "SMAL" in dataset_name.upper():
        return "SMAL"

    if "DFAUST" in dataset_name.upper():
        return "DFAUST"

    raise Exception(f"Invalid dataset '{dataset_name}'")