"""Constants used in scripts."""

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

TEST_KEYS = [
    202013449349300126,
    202013449349300131,
    202013449349300201,
    202013449349300206,
    202013449349300211,
    202013449349300216,
    202013449349300221,
    202013449349300226,
    202013449349300231,
    202013449349300236,
    202013449349300241,
    202013449349300246,
    201403169349305000,
    201403179349100000,
    201403159349201000,
]
