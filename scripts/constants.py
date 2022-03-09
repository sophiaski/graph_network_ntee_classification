"""Constants used in scripts."""

# Model variables
SEED = 42
EXPERIMENT_KEYS = [
    ["broad", "ntee"],
    ["stratify_sklearn", "stratify_none"],
    ["train", "validation", "test"],
]

# method: grid, random, bayesian
# Random reveals what values are not very good, then reduce space to do grid search

parameter_dict = {
    "optimizer": {"values": ["adam", "sgd"]},
    "fc_layer_size": {"values": [128, 256, 512]},
    "dropout": {"values": [0.3, 0.4, 0.5]},
    "epochs": {"values": [2, 3, 4]},
    "max_length": {"values": [64, 128]},
    "batch_size": {"values": [8, 16, 32]},
    "lr": {"values": [2e-05, 3e-05, 5e-05]},
}
sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": parameter_dict,
}

# the header rows as they'll appear in the output
HEADERS_990 = [
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
