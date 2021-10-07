from scripts.python.pheno.datasets.fields import Field


def get_columns_dict(dataset: str):
    d = None
    if dataset == "GSE152027":
        d = {
            'Status': 'status',
            'Age': 'ageatbloodcollection',
            'Sex': 'gender',
        }
    elif dataset == "GSE145361":
        d = {
            'Status': 'disease-state',
            'Age': None,
            'Sex': 'gender',
        }
    elif dataset == "GSE111629":
        d = {
            'Status': 'disease state',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE72774":
        d = {
            'Status': 'diseasestatus',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE84727":
        d = {
            'Status': 'disease_status',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE80417":
        d = {
            'Status': 'disease status',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE125105":
        d = {
            'Status': 'diagnosis',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE113725":
        d = {
            'Status': 'groupid',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE89353":
        d = {
            'Status': 'Status',
            'Age': None,
            'Sex': 'gender',
        }
    elif dataset == "GSE53740":
        d = {
            'Status': 'diagnosis',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE156994":
        d = {
            'Status': 'sample_group',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE144858":
        d = {
            'Status': 'disease state',
            'Age': 'age',
            'Sex': 'Sex',
        }


    elif dataset == "GSE42861":
        d = {
            'Status': 'subject',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE147221":
        d = {
            'Status': 'status',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE168739":
        d = {
            '': '',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE128235":
        d = {
            'Status': 'diagnosis',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE87648":
        d = {
            'Status': 'simplified_diagnosis',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE106648":
        d = {
            'Status': 'disease status',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSEUNN":
        d = {
            'Status': 'Status',
            'Age': 'Age',
            'Sex': 'Sex',
        }
    return d


def get_column_name(dataset: str, feature: str):
    d = get_columns_dict(dataset)
    return d[feature]


def get_status_dict(dataset: str):
    d = None
    if dataset == "GSE152027":
        d = {"Control": [Field('Control', 'CON')], "Cases": [Field('Schizophrenia', 'SCZ'), Field('First episode psychosis', 'FEP')]}
    elif dataset == "GSE145361":
        d = {"Control": [Field('Control', 'Control')], "Cases": [Field('Parkinson', "Parkinson's disease")]}
    elif dataset == "GSE111629":
        d = {"Control": [Field('Control', "PD-free control")], "Case": [Field('Parkinson', "Parkinson's disease (PD)")]}
    elif dataset == "GSE72774":
        d = {"Control": [Field('Control', "control")], "Case": [Field('Parkinson', "PD")]}
    elif dataset == "GSE84727":
        d = {"Control": [Field('Control', 1)], "Case": [Field('Schizophrenia', 2)]}
    elif dataset == "GSE80417":
        d = {"Control": [Field('Control', 1)], "Case": [Field('Schizophrenia', 2)]}
    elif dataset == "GSE125105":
        d = {"Control": [Field('Control', "control")], "Case": [Field('Depression',"case")]}
    elif dataset == "GSE113725":
        d = {"Control": [Field('Control', 4)], "Case": [Field('Depression', 2), Field('Inflammatory disorder', 3)]}
    elif dataset == "GSE89353":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field('Intellectual disability and congenital anomalies', 'IDCA')]}
    elif dataset == "GSE53740":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field('Progressive supranuclear palsy', 'PSP'), Field('Frontotemporal dementia', 'FTD')]}
    elif dataset == "GSE156994":
        d = {"Control": [Field('Control', 'CTRL')], "Case": [Field('Sporadic Creutzfeldt-Jakob disease', 'sCJD')]}
    elif dataset == "GSE144858":
        d = {"Control": [Field('Control', 'control')], "Case": [Field('Alzheimer', "Alzheimer's disease"), Field('Mild cognitive impairment', "mild cognitive impairment")]}


    elif dataset == "GSE42861":
        d = {"Control": "Normal", "Cases": ["Patient"]}
    elif dataset == "GSE147221":
        d = {"Control": "Control", "Case": "Case"}
    elif dataset == "GSE152027":
        d = {"Control": "CON", "Case": "SCZ"}
    elif dataset == "GSE168739":
        d = {"Control": "", "": ""}
    elif dataset == "GSE111629":
        d = {"Control": "PD-free control", "Case": "Parkinson's disease (PD)"}
    elif dataset == "GSE128235":
        d = {"Control": "control", "Case": "case"}
    elif dataset == "GSE87648":
        d = {"Control": "HL", "Case": "CD"}
    elif dataset == "GSE144858":
        d = {"Control": "control", "Case": "Alzheimer's disease"}
    elif dataset == "GSE106648":
        d = {"Control": "Healthy control", "Case": "MS case"}
    elif dataset == "GSEUNN":
        d = {"Control": "Control", "Case": "ESRD"}
    return d


def get_statuses_datasets_dict():
    d = {
        'Schizophrenia': ['GSE152027', 'GSE84727', 'GSE80417'],
        'First episode psychosis': ['GSE152027'],
        'Parkinson': ['GSE145361', 'GSE111629', 'GSE72774'],
        'Depression': ['GSE125105', 'GSE113725'],
        'Intellectual disability and congenital anomalies': ['GSE89353'],
        'Progressive supranuclear palsy': ['GSE53740'],
        'Frontotemporal dementia': ['GSE53740'],
        'Sporadic Creutzfeldt-Jakob disease': ['GSE156994'],
        'Mild cognitive impairment': ['GSE144858'],
        'Alzheimer': ['GSE144858']
    }
    return d


def get_status_names_dict22(dataset: str):
    d = {
        'GSE152027': {"Control": ["Control"], "Cases": ['Schizophrenia', 'First episode psychosis']},
        'GSE42861': {"Control": "Control", "Case": 'Rheumatoid Arthritis'},
        'GSE80417': {"Control": "Control", "Case": 'Schizophrenia'},
        'GSE84727': {"Control": "Control", "Case": 'Schizophrenia'},
        'GSE125105': {"Control": "Control", "Case": 'Depression'},
        'GSE147221': {"Control": "Control", "Case": 'Schizophrenia'},
        'GSE168739': {"Control": "Control", "Case": ''},
        'GSE111629': {"Control": "Control", "Case": "Parkinson's disease"},
        'GSE128235': {"Control": "Control", "Case": 'Depression'},
        'GSE72774': {"Control": "Control", "Case": "Parkinson's disease"},
        'GSE53740': {"Control": "Control", "Case": 'Frontotemporal dementia'},
        'GSE87648': {"Control": "Control", "Case": "Crohn's disease"},
        'GSE144858': {"Control": "Control", "Case": "Alzheimer's disease"},
        'GSE106648': {"Control": "Control", "Case": "Multiple Sclerosis"},
        'GSE156994': {"Control": "Control", "Case": "Sporadic Creutzfeldt-Jakob disease"},
        'GSEUNN': {"Control": "Control", "Case": "ESRD"},
    }
    return d[dataset]


def get_sex_dict(dataset: str):
    d = None
    if dataset == "GSE152027":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE145361":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE111629":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE72774":
        d = {"F": "female", "M": "male"}
    elif dataset == "GSE84727":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE80417":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE125105":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE113725":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE89353":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE53740":
        d = {"F": "FEMALE", "M": "MALE"}
    elif dataset == "GSE156994":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE144858":
        d = {"F": "Female", "M": "Male"}


    elif dataset == "GSE42861":
        d = {"F": "f", "M": "m"}
    elif dataset == "GSE147221":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE168739":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE128235":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE87648":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE106648":
        d = {"F": "female", "M": "male"}
    elif dataset == "GSEUNN":
        d = {"F": "F", "M": "M"}
    return d
