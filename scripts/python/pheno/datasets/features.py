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
    elif dataset == "GSEUNN":
        d = {
            'Status': 'Status',
            'Age': 'Age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE87571":
        d = {
            'Status': 'Sample_Group',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE74193":
        d = {
            'Status': 'group',
            'Age': 'age (in years)',
            'Sex': 'sex (clinical gender)',
        }
    elif dataset == "GSE48325":
        d = {
            'Status': 'group',
            'Age': 'age',
            'Sex': 'sex (1 - male, 2 - female)',
        }
    elif dataset == "GSE61258":
        d = {
            'Status': 'diseasestatus',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE61446":
        d = {
            'Status': 'subject status',
            'Age': 'subject age',
            'Sex': 'gender',
        }
    elif dataset == "GSE152026":
        d = {
            'Status': 'phenotype',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE164056":
        d = {
            'Status': 'status',
            'Age': 'age',
            'Sex': 'Sex',
        }
    elif dataset == "GSE52588":
        d = {
            'Status': 'group',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE41169":
        d = {
            'Status': 'diseasestatus (1=control, 2=scz patient)',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE116378":
        d = {
            'Status': 'group',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE116379":
        d = {
            'Status': 'group',
            'Age': 'age',
            'Sex': 'gender',
        }
    elif dataset == "GSE40279":
        d = {
            'Status': 'Status',
            'Age': 'age (y)',
            'Sex': 'gender',
        }
    elif dataset == "GSE55763":
        d = {
            'Status': 'characteristics_ch1.1.dataset',
            'Age': 'characteristics_ch1.3.age',
            'Sex': 'characteristics_ch1.2.gender',
        }

    return d


def get_column_name(dataset: str, feature: str):
    d = get_columns_dict(dataset)
    return d[feature]


def get_status_dict(dataset: str):
    d = None
    if dataset == "GSE152027":
        d = {"Control": [Field('Control', 'CON')], "Case": [Field('Schizophrenia', 'SCZ'), Field('First episode psychosis', 'FEP')]}
    elif dataset == "GSE145361":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field('Parkinson', "Parkinson's disease")]}
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
    elif dataset == "GSEUNN":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field("ESRD", "ESRD")] }
    elif dataset == "GSE87571":
        d = {"Control": [Field('Control', 'C')]}
    elif dataset == "GSE74193":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field('Schizophrenia', 'Schizo')]}
    elif dataset == "GSE48325":
        d = {"Control": [Field('Control', 'NORMAL'), Field('Healthy Obese', 'Healthy Obese')], "Case": [Field('Non-alcoholic fatty liver disease', 'NAFLD'), Field('Non-alcoholic steatohepatitis', 'NASH')]}
    elif dataset == "GSE61258":
        d = {"Control": [Field('Control', 'Control'), Field('Healthy Obese', 'HealthyObese')], "Case": [Field('Non-alcoholic fatty liver disease', 'NAFLD'), Field('Non-alcoholic steatohepatitis', 'NASH'), Field('Primary biliary cholangitis', 'PBC'), Field('Primary sclerosing cholangitis', 'PSC')]}
    elif dataset == "GSE61446":
        d = {"Control": [Field('Healthy Obese', 'severely obese patient')]}
    elif dataset == "GSE152026":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field('First episode psychosis', 'Case')]}
    elif dataset == "GSE164056":
        d = {"Control": [Field('Control', 'Control')], "Case": [Field('Case', 'Case')]}
    elif dataset == "GSE52588":
        d = {"Control": [Field('Siblings', 'Siblings'), Field('Mothers', 'Mothers')], "Case": [Field('Down syndrome', 'DS')]}
    elif dataset == "GSE41169":
        d = {"Control": [Field('Control', 1)], "Case": [Field('Schizophrenia', 2)]}
    elif dataset == "GSE116378":
        d = {"Control": [Field('Control', 'CTR')], "Case": [Field('Schizophrenia', 'SCZ')]}
    elif dataset == "GSE116379":
        d = {"Control": [Field('Control', 'CTR_Non_Famine'), Field('Control', 'CTR_Famine')], "Case": [Field('Schizophrenia', 'SCZ_Non_Famine'), Field('Schizophrenia', 'SCZ_Famine')]}
    elif dataset == "GSE40279":
        d = {"Control": [Field('Control', 'C')]}
    elif dataset == "GSE55763":
        d = {"Control": [Field('Control', 'population study'), Field('Duplicate0', 'population study; technical replication study'), Field('Duplicate1', 'technical replication study')]}

    return d

def get_default_statuses_ids(dataset: str):
    statuses = None
    if dataset == "GSE152027":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE145361":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE111629":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE72774":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE84727":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE80417":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE125105":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE113725":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE89353":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE53740":
        statuses = {"Control": [0], "Case": [1]}
    elif dataset == "GSE156994":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE144858":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSEUNN":
        statuses = {"Control": [0], "Case": [0]}
    elif dataset == "GSE87571":
        statuses = {"Control": [0]}
    elif dataset == "GSE74193":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE48325":
        statuses = {"Control": [0, 1], 'Case': [0, 1]}
    elif dataset == "GSE61258":
        statuses = {"Control": [0, 1], 'Case': [0, 1, 2, 3]}
    elif dataset == "GSE61446":
        statuses = {"Control": [0]}
    elif dataset == "GSE152026":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE164056":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE52588":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE41169":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE116378":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE116379":
        statuses = {"Control": [0], 'Case': [0]}
    elif dataset == "GSE40279":
        statuses = {"Control": [0]}
    elif dataset == "GSE55763":
        statuses = {"Control": [0]}

    return statuses


def get_default_statuses(dataset: str):
    status_dict = get_status_dict(dataset)
    default_statuses_ids = get_default_statuses_ids(dataset)
    statuses = []
    for part, indices in default_statuses_ids.items():
        for i in indices:
            statuses.append(status_dict[part][i].label)
    return statuses


def get_status_dict_default(dataset: str):
    status_dict = get_status_dict(dataset)
    default_statuses_ids = get_default_statuses_ids(dataset)
    status_dict_default = {}
    for status, fields in status_dict.items():
        default_fields = [fields[def_id] for def_id in default_statuses_ids[status]]
        status_dict_default[status] = default_fields
    return status_dict_default


def get_statuses_datasets_dict():
    d = {
        'Schizophrenia': ['GSE152027', 'GSE84727', 'GSE80417'],
        'First episode psychosis': ['GSE152027'],
        'Parkinson': ['GSE145361', 'GSE111629', 'GSE72774'],
        #'Depression': ['GSE125105', 'GSE113725'],
        'Depression': ['GSE125105'],
        'Intellectual disability and congenital anomalies': ['GSE89353'],
        'Progressive supranuclear palsy': ['GSE53740'],
        'Frontotemporal dementia': ['GSE53740'],
        'Sporadic Creutzfeldt-Jakob disease': ['GSE156994'],
        'Mild cognitive impairment': ['GSE144858'],
        'Alzheimer': ['GSE144858'],
        'Healthy': ['GSE87571', 'GSE40279']
    }
    return d


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
    elif dataset == "GSEUNN":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE87571":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE74193":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE48325":
        d = {"F": 2, "M": 1}
    elif dataset == "GSE61258":
        d = {"F": "female", "M": "male"}
    elif dataset == "GSE61446":
        d = {"F": "female", "M": "male"}
    elif dataset == "GSE152026":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE164056":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE52588":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE41169":
        d = {"F": "Female", "M": "Male"}
    elif dataset == "GSE116378":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE116379":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE40279":
        d = {"F": "F", "M": "M"}
    elif dataset == "GSE55763":
        d = {"F": "F", "M": "M"}

    return d
