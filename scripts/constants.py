"""Constants used in scripts."""

# Model experiments set-up
SEED = 117

EXPERIMENT_KEYS = [
    ["broad", "ntee"],
    ["sklearn", "none"],
    ["train", "valid", "test", "unlabeled"],
]
SWEEP_INIT = {
    "optimizer": "adam",
    "learning_rate": 5e-05,
    "epochs": 1,
    "batch_size": 16,
    "classifier_dropout": 0.3,
    "perc_warmup_steps": 0.25,
    "max_length": 256,
    "clip_grad": True,
    "frac": 0.25,
    "complex_graph": True,
    "add_more_targets": True,
}

FINAL_CONFIG = {
    "ntee": {
        "cat_type": "ntee",
        "optimizer": "adam",
        "learning_rate": 5e-05,
        "epochs": 1,
        "batch_size": 16,
        "classifier_dropout": 0.3,
        "perc_warmup_steps": 0.25,
        "max_length": 256,
        "clip_grad": True,
        "frac": 1.0,
        "pretrained_model_path": None,
        "strat_type": "sklearn",
        "sampler": "weighted_norm",
        "complex_graph": True,
        "add_more_targets": False,
    },
    "broad": {
        "cat_type": "broad",
        "optimizer": "adam",
        "learning_rate": 5e-05,
        "epochs": 1,
        "batch_size": 16,
        "classifier_dropout": 0.3,
        "perc_warmup_steps": 0.25,
        "max_length": 256,
        "clip_grad": True,
        "frac": 1.0,
        "pretrained_model_path": None,
        "strat_type": "sklearn",
        "sampler": "weighted_norm",
        "complex_graph": True,
        "add_more_targets": False,
    },
}

# Very broad random search
SWEEP_CONFIG_Mar12 = {
    "method": "random",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "optimizer": {"values": ["adam"]},
        "learning_rate": {"values": [0.001, 0.0001, 0.00002, 0.00003, 0.00005]},
        "epochs": {"values": [2, 3]},
        "classifier_dropout": {"values": [0.1, 0.3, 0.5]},
        "batch_size": {"values": [8, 16, 32]},
        "perc_warmup_steps": {"values": [0, 0.01, 0.1, 0.25]},
        "clip_grad": {"values": [True, False]},
        "max_length": {"values": [64, 128, 256]},
        "frac": {"values": [0.25, 0.5, 0.75]},
    },
}

# More constrained random search
SWEEP_CONFIG_Mar13 = {
    "method": "random",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "optimizer": {"values": ["adam"]},
        "classifier_dropout": {"values": [0.1, 0.3, 0.5]},
        "learning_rate": {"values": [0.00003, 0.00005]},
        "epochs": {"values": [1, 2]},
        "batch_size": {"values": [8, 16, 32]},
        "perc_warmup_steps": {"values": [0.1, 0.25]},
        "clip_grad": {"values": [False]},
        "max_length": {"values": [64]},
        "frac": {"values": [0.7, 0.9]},
        "replacement": {"values": [True]},
    },
}

# Final baseline random search, switching to maximizing accuracy
SWEEP_CONFIG_Mar14 = {
    "method": "random",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "optimizer": {"values": ["adam"]},
        "classifier_dropout": {"values": [0.1, 0.3, 0.4]},
        "learning_rate": {"values": [0.00002, 0.00003, 0.00005]},
        "epochs": {"values": [1]},
        "batch_size": {"values": [16, 32, 64]},
        "perc_warmup_steps": {"values": [0, 0.25, 0.5]},
        "clip_grad": {"values": [False]},
        "max_length": {"values": [64]},
        "frac": {"values": [1.0]},
    },
}

# Final baseline grid search, switching to maximizing accuracy
# SWEEP_CONFIG_Mar15
SWEEP_CONFIG_Mar15 = {
    "method": "random",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "optimizer": {"values": ["adam"]},
        "classifier_dropout": {"values": [0.1, 0.2, 0.3, 0.4]},
        "learning_rate": {"values": [0.00005, 0.00003, 0.00002]},
        "epochs": {"values": [2, 3]},
        "batch_size": {"values": [16, 32, 64]},
        "perc_warmup_steps": {"values": [0, 0.1]},
        "clip_grad": {"values": [False]},
        "max_length": {"values": [64]},
        "frac": {"values": [1.0]},
    },
}

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "optimizer": {"value": "adam"},
        "classifier_dropout": {"value": 0.3},
        "learning_rate": {"value": 0.00005},
        "epochs": {"value": 2},
        "batch_size": {"value": 16},
        "perc_warmup_steps": {"value": 0.2},
        "clip_grad": {"value": True},
        "max_length": {"value": 256},
        "frac": {"value": 1.0},
        "complex_graph": {"values": [True, False]},
        "add_more_targets": {"values": [True, False]},
    },
}

# The header rows as they'll appear in the output
IRSX_HEADERS = [
    "object_id",
    "ein",
    "form_type",
    "submission_ts",
    "business_name",
    "tax_period_begin",
    "tax_period_end",
    "tax_year",
    "formation_year",
    "mission_descript",
    "mission_descript_2",
    "mission_descript_3",
    "mission_descript_4",
    "mission_descript_5",
    "program_descript",
    "program_descript_2",
    "program_descript_3",
    "program_descript_4",
]

GRANTS_HEADERS = [
    "grantee_ein",
    "grantee",
    "grant_desc",
    "cash_grant_amt",
    "grantor",
    "grantor_ein",
    "tax_period",
    "grantee_city",
    "grantee_state",
    "grantee_zipcode",
    "grantor_city",
    "grantor_state",
    "grantor_zipcode",
    "grantor_location",
    "grantee_location",
    "grantor_info",
    "grantee_info",
]

BENCHMARK_HEADERS = [
    "ein",
    "taxpayer_name",
    "return_type",
    "sequence",
    "NTEE1",
    "broad_cat",
]

# Available schedule types in IRSx, not used in scripts, just FYI
SCHEDULES_990 = [
    "ReturnHeader990x",
    "IRS990",
    "IRS990EZ",
    "IRS990PF",
    "IRS990ScheduleA",
    "IRS990ScheduleB",
    "IRS990ScheduleC",
    "IRS990ScheduleD",
    "IRS990ScheduleG",
    "IRS990ScheduleH",
    "IRS990ScheduleI",
    "IRS990ScheduleJ",
    "IRS990ScheduleK",
    "IRS990ScheduleL",
    "IRS990ScheduleM",
    "IRS990ScheduleO",
    "IRS990ScheduleR",
]

WANT_SKED_GROUPS = {
    "ReturnHeader990x": False,
    "IRS990": False,
    "IRS990EZ": True,
    "IRS990PF": True,
}

WANT_SKED_PARTS = {
    "ReturnHeader990x": True,
    "IRS990": True,
    "IRS990EZ": True,
    "IRS990PF": True,
}

BROAD_CAT_DICT = {
    "I": ["A"],
    "II": ["B"],
    "III": ["C", "D"],
    "IV": ["E", "F", "G", "H"],
    "V": ["I", "J", "K", "L", "M", "N", "O", "P"],
    "VI": ["Q"],
    "VII": ["R", "S", "T", "U", "V", "W"],
    "VIII": ["X"],
    "IX": ["Y"],
    "X": ["Z"],
}

BROAD_CAT_NAME = {
    "I": "Arts, Culture, and Humanities",
    "II": "Education",
    "III": "Environment and Animals",
    "IV": "Health",
    "V": "Human Services",
    "VI": "International, Foreign Affairs",
    "VII": "Public, Societal Benefit",
    "VIII": "Religion Related",
    "IX": "Mutual/Membership Benefit",
    "X": "Unknown, Unclassified",
}

NTEE_NAME = {
    "A": "Arts, Culture and Humanities",
    "B": "Educational Institutions and Related Activities",
    "C": "Environmental Quality, Protection and Beautification",
    "D": "Animal-Related",
    "E": "Health – General and Rehabilitative",
    "F": "Mental Health, Crisis Intervention",
    "G": "Diseases, Disorders, Medical Disciplines",
    "H": "Medical Research",
    "I": "Crime, Legal-Related",
    "J": "Employment, Job-Related",
    "K": "Food, Agriculture and Nutrition",
    "L": "Housing, Shelter",
    "M": "Public Safety, Disaster Preparedness and Relief",
    "N": "Recreation, Sports, Leisure, Athletics",
    "O": "Youth Development",
    "P": "Human Services – Multipurpose and Other",
    "Q": "International, Foreign Affairs and National Security",
    "R": "Civil Rights, Social Action, Advocacy",
    "S": "Community Improvement, Capacity Building",
    "T": "Philanthropy, Voluntarism and Grantmaking Foundations",
    "U": "Science and Technology Research Institutes, Services",
    "V": "Social Science Research Institutes, Services",
    "W": "Public, Society Benefit – Multipurpose and Other",
    "X": "Religion-Related, Spiritual Development",
    "Y": "Mutual/Membership Benefit Organizations, Other",
    "Z": "Unknown, Unclassified",
}

# Graph! Simple
NODE_COLS = [
    "ein",
    "node_type",
    "sequence",
    "NTEE1",
    "broad_cat",
    "benchmark_status",
]
EDGE_COLS = ["src", "dst", "edge_type", "tax_period", "cash_grant_amt"]

# Graph! Complex
NODE_COLS_COMPLEX = []
EDGE_COLS_COMPLEX = []
