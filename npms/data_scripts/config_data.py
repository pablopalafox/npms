
identities_mixamo = [
    "adam",
    "alex",
    "alien",
    "astra",
    "brian",
    "bryce",
    "chad",
    "crypto",
    "david",
    "douglas",
    "elizabeth",
    "eve",
    "james",
    "jennifer",
    "jody",
    "joe",
    "josh",
    "kate",
    "leonard",
    "lewis",
    "louise",
    "mannequin",
    "martha",
    "megan",
    "ninja",
    "olivia",
    "pete",
    "racer",
    "regina",
    "romero",
    "roth",
    "shae",
    "shannon",
    "sophie",
    "steve",
    "suzie",
    "swatguy",
]

identities_augmented = [
    "arissa",
    "brute",
    "ely",
    "exo",
    "liam",
    "malcom",
    "markerman",
    "pete_no_helmet",
    "racer_jody",
    "remy",
]

test_identities_mixamo = [
    'alien',
    'joe',
    'olivia',
    'sophie',
]

test_animations_mixamo_by_identity = {
    "alien": [
        'female_hip_hop_slide_step_dancing',
    ],
    "joe": [
        'breakdance_brooklyn_style_uprocking',
    ],
    "olivia": [
        'female_samba_ijexa_break',
    ],
    "sophie": [
        'female_salsa_dancing',
    ],
}
##########################################################
##########################################################

##########################################################
# AMASS
##########################################################
test_identities_amass = [
    'Transitionsmocap_s003',
    'SSMsynced_s000',
    'SSMsynced_s001',
    'SSMsynced_s002',
]
##########################################################
##########################################################

##########################################################
# MANO
##########################################################
test_identities_mano = [
    'test_id_24_38l_mirrored',
    'test_id_30_42l_mirrored',
    'test_id_39_26l_mirrored',
    'test_id_50_43r',
    'test_id_10_45l_mirrored',
]
##########################################################
##########################################################

##########################################################
# CAPE
##########################################################
test_identities_cape = [
    # 00032
    '00032_longshort', # doesn't have scans, but we cannot use it in the train set of course
    '00032_shortlong',
    '00032_shortshort',
    # 03223
    '03223_longshort', # doesn't have scans, but we cannot use it in the train set of course
    '03223_shortlong',
    '03223_shortshort'
]

animations_with_scan_by_identity = {
    #####################
    ####### 00032 #######
    #####################
    "00032_shortlong": [
        'hips',
        'shoulders_mill',
        'tilt_twist_left',
    ],
    "00032_shortshort": [
        'hips',
        'shoulders_mill',
    ],
    #####################
    ####### 00096 #######
    #####################
    "00096_shortlong": [
        'hips',
        'shoulders_mill',
    ],
    "00096_shortshort": [
        'hips',
        'shoulders_mill',
        'tilt_twist_left',
    ],
    #####################
    ####### 00159 #######
    #####################
    "00159_shortlong": [
        'pose_model',
        'tilt_twist_left',
    ],
    "00159_shortshort": [
        'pose_model',
        'tilt_twist_left',
    ],
    #####################
    ####### 03223 #######
    #####################
    "03223_shortlong": [
        'hips',
        'shoulders_mill',
        'tilt_twist_left',
    ],
    "03223_shortshort": [
        'hips',
        'shoulders_mill',
        'tilt_twist_left',
    ],
}

test_animations_cape_by_identity = {
    #####################
    ####### 00032 #######
    #####################
    "00032_shortlong": [
        'tilt_twist_left'
    ],
    "00032_shortshort": [
        'shoulders_mill',
    ],
    #####################
    ####### 03223 #######
    #####################
    "03223_shortlong": [
        # 'shoulders_mill',
        'tilt_twist_left'
    ],
    "03223_shortshort": [
        'tilt_twist_left',
    ],
}

##########################################################
# DFAUST
##########################################################
test_animations_dfaust_by_identity = {
    "50021": [
        "chicken_wings",
        "knees",
        "one_leg_jump",
        "punching",
        "shake_arms",
        "shake_shoulders",
        "hips",
        "light_hopping_stiff",
        "one_leg_loose",
        "running_on_spot",
        "shake_hips",
    ],
}

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

animations_by_style = {
    "hip_hop": [
        "Bboy Hip Hop Variation One",
        "Bboy Uprock Start",
        "Bboy Pose To Standing Idle",
        "Bboy Uprock Indian Step",
        "Booty Step Hip Hop Dance",
        "Female Hip Hop 'Arm Wave' Dancing",
        "Female Hip Hop 'Body Wave' Dancing",
        "Female Hip Hop 'Kick Step' Dancing",
        "Female Hip Hop 'Rib Pops' Dancing",
        "Female Hip Hop 'Running Man' Dancing",
        "Female Hip Hop 'Slide Step' Dancing",
        "Hip Hop Dance Moonwalk",
        "Hip Hop Dancing Bboy Variation One",
        "Hip Hop Dancing Really Twirl",
        "Hip Hop Dancing Shimmy",
        "Hip Hop Dancing Side To Side",
        "Hip Hop Just Listening Dancing Variation",
        "Hip Hop Quake Variation 1",
        "Hip Hop Robot Dance Variation Two",
        "Hip Hop Runningman Dance",
        "Locking Hip Hop Dance Variation Two",
        "Robot Hip Hop Dance Variation One",
        "Slide Hip Hop Dance Variation One",
        "Slide Hip Hop Walk",
        "Snake Hip Hop Dance",
        "Step Hip Hop Dance",
        "Tut Hip Hop Dance Variation One",
        "Tut Hip Hop Dance Variation Two",
        "Wave Hip Hop Dance Variation One",
        "Wave Hip Hop Dance Variation Two",
        "Waving The Arms Hip Hop Dance",
    ],
    "silly": [
        "Shopping Cart Dance",
        "Silly Dancing The Cabbage Patch",
        "Silly Dancing The Twist",
        "Silly Run",
        "Turn To Silly Run Away",
    ],
    "rumba": [
        "Female Rumba Dancing -  Loop"
    ],
    "breakdance": [
        "Breakdance Arch Freeze Var 3",
        "Breakdance Arch Freeze Var 4",
        "Breakdance Crossleg Freeze Transition To Stand",
        "Breakdance Finishing Combo Var 1",
        "Breakdance Finishing Combo Var 2",
        "Breakdance Finishing Combo Var 3",
        "Breakdance Flair",
        "Breakdance Flair End",
        "Breakdance Brooklyn Style Uprocking",
        "Breakdance Flair Start",
        "Breakdance Freeze Combo",
        "Breakdance Ground Footwork To Crossleg Freeze",
        "Breakdance Ground Footwork To Hands On Hips Idle",
        "Breakdance Ground Footwork To Stand Idle",
        "Breakdance Ground Footwork Var 1- Loop",
        "Breakdance Ground Footwork Var 2",
        "Breakdance Ground Footwork Var 3",
        "Breakdance Head Spin",
        "Breakdance Headspin End",
        "Breakdance Headspin Start",
        "Breakdance Single Handstand Freeze Var 1",
        "Breakdance Single Handstand Freeze Var 2",
        "Breakdance Single Handstand Spin End",
        "Breakdance Single Handstand Spin Start",
        "Breakdance Uprock",
        "Breakdance Uprock Footwork Var. 2",
        "Breakdance Uprock Start",
        "Breakdance Uprock Transition To Ground Moves",
        "Breakdance Uprock Var. 1",
        "Ready To Breakdance End",
        "Ready To Breakdance Start",
        "Start Breakdance Swipes Maneuver",
    ],
    "capoeira": [
        "Capoeira Cartwheel Move",
        "Capoeira Cartwheel To Roll Escape",
        "Capoeira Flip Escape",
        "Capoeira Floor Escape",
        "Capoeira Forward Kick",
        "Capoeira Ground Spin Kick",
        "Capoeira Groundwork",
        "Capoeira High Kick With Forward Step",
        "Capoeira Idle",
        "Capoeira Kicks",
        "Capoeira Retreating Spin Kick",
        "Capoeira Side Kick",
        "Capoeira Side Step Flip Escape",
        "Capoeira Side Step Spin Kick",
        "Capoeira Spin Kick To Escape",
        "Capoeira Step Spin Kick",
        "Capoeira Step To Cartwheel",
        "Capoeira Thrust Kick",
    ],
    "swing": [
        "Dance Swing Charleston",
        "Swing Dance Charleston \"Crazy Legs\"",
        "Swing Dance Charleston Variation 2",
        "Swing Dance Shim Sham Variation 1",
        "Swing Dance Shim Sham Variation 2",
    ],
    "samba": [
        "Afoxe Samba Reggae Dance",
        "Defile De Samba Parade Variation",
        "Female Fast Samba Enredo Variation Two",
        "Female Samba Ijexa Break",
        "Female Samba Pagode Variation Five- Loop",
        "Samba Dance Olodum Variation Two",
        "Samba Funky Pocoto Variation 1",
        "Samba Gandy Variation 1",
    ],
    "house": [
        "House Dance Variation 3",
        "House Dance Variation Four",
        "House Dance Variation Two",        
    ],
    "african": [
        "Female \"African Noodle\" Dance",
        "Female \"African Rainbow\" Dance",
    ],
    "soul": [
        "Basic Northern Soul Step",
        "Northern Soul Dance Spin",
        "Northern Soul Dance Spin And Floor Work",
        "Northern Soul Dance Spin Dip And Splits",
        "Northern Soul Dance Spin On Floor",
    ],
    "salsa": [
        "Female Salsa Dancing",
        "Male Partner Salsa Variation Five",
        "Male Partner Salsa Variation One",
        "Male Partner Salsa Variation Two",
        "Male Salsa Variation Eight",
        "Salsa Dance Variation Five",
        "Salsa Dancing Double Twirl",
        "Salsa Dancing Side To Side",
        "Salsa Dancing Twirl And Clap",
    ],
    "thriller": [
        "Thriller Dance Part 1",
        "Thriller Dance Part 2",
        "Thriller Dance Part 3",
        "Thriller Dance Part 4",
    ],
    "jazz": [
        "Female Jazz Dancing 'Around The World'",
        "Female Jazz Dancing 'Around The World'- Loop",
        "Female Jazz Dancing 'Rockette Kick'",
        "Female Jazz Square Dancing",
    ],
    "other": [
        "Baseball Bunt Swing",
        "Baseball Swinging Strike",
        "Bellydance Variation 2",
        "Dancing The Twerk",
        "Doing The Can Can",
        "Doing The Chicken Dance",
        "Doing The Hokey Pokey",
        "Doing The Shuffling Dance",
        "Doing The Twist Dance",
        "Doing The Ymca Dance",
        "Male Walking With Shopping Bag",
        "Russian Kettlebell Swing",
        "The Popular K-Pop Dance",
        "Walking Backwards Sliding Feet",
        "Watering With A Hose",
    ],
}

################################################################################
################################################################################
################################################################################
################################################################################

animation_info = {
    # Breakdance
    "bboy_hip_hop_variation_one": {
        "num_frames": 53,
        "product_name": "Bboy Hip Hop Variation One",
    },
    "bboy_uprock_start": {
        "num_frames": 44,
        "product_name": "Bboy Uprock Start",
    },
    "bboy_pose_to_standing_idle": {
        "num_frames": 56,
        "product_name": "Bboy Pose To Standing Idle",
    },
    "bboy_uprock_indian_step": {
        "num_frames": 56,
        "product_name": "Bboy Uprock Indian Step",
    },
    "booty_step_hip_hop_dance": {
        "num_frames": 118,
        "product_name": "Booty Step Hip Hop Dance",
    },
    "female_hip_hop_arm_wave_dancing": {
        "num_frames": 125,
        "product_name": "Female Hip Hop 'Arm Wave' Dancing",
    },
    "female_hip_hop_body_wave_dancing": {
        "num_frames": 170,
        "product_name": "Female Hip Hop 'Body Wave' Dancing",
    },
    "female_hip_hop_kick_step_dancing": {
        "num_frames": 152,
        "product_name": "Female Hip Hop 'Kick Step' Dancing",
    },
    "female_hip_hop_rib_pops_dancing": {
        "num_frames": 81,
        "product_name": "Female Hip Hop 'Rib Pops' Dancing",
    },
    "female_hip_hop_running_man_dancing": {
        "num_frames": 114,
        "product_name": "Female Hip Hop 'Running Man' Dancing",
    },
    "female_hip_hop_slide_step_dancing": {
        "num_frames": 108,
        "product_name": "Female Hip Hop 'Slide Step' Dancing",
        "train": False
    },
    "hip_hop_dance_moonwalk": {
        "num_frames": 414,
        "product_name": "Hip Hop Dance Moonwalk",
        "train": False
    },
    "hip_hop_dancing_bboy_variation_one": {
        "num_frames": 331,
        "product_name": "Hip Hop Dancing Bboy Variation One",
    },
    "hip_hop_dancing_really_twirl": {
        "num_frames": 376,
        "product_name": "Hip Hop Dancing Really Twirl",
    },
    "hip_hop_dancing_shimmy": {
        "num_frames": 288,
        "product_name": "Hip Hop Dancing Shimmy",
    },
    "hip_hop_dancing_side_to_side": {
        "num_frames": 328,
        "product_name": "Hip Hop Dancing Side To Side",
    },
    "hip_hop_just_listening_dancing_variation": {
        "num_frames": 446,
        "product_name": "Hip Hop Just Listening Dancing Variation",
    },
    "hip_hop_quake_variation_1": {
        "num_frames": 379,
        "product_name": "Hip Hop Quake Variation 1",
        "train": False
    },
    "hip_hop_robot_dance_variation_two": {
        "num_frames": 368,
        "product_name": "Hip Hop Robot Dance Variation Two",
        "train": False
    },
    "hip_hop_runningman_dance": {
        "num_frames": 147,
        "product_name": "Hip Hop Runningman Dance",
    },
    "locking_hip_hop_dance_variation_two": {
        "num_frames": 408,
        "product_name": "Locking Hip Hop Dance Variation Two",
    },
    "robot_hip_hop_dance_variation_one": {
        "num_frames": 371,
        "product_name": "Robot Hip Hop Dance Variation One",
    },
    "slide_hip_hop_dance_variation_one": {
        "num_frames": 416,
        "product_name": "Slide Hip Hop Dance Variation One",
    },
    "slide_hip_hop_walk": {
        "num_frames": 24,
        "product_name": "Slide Hip Hop Walk",
    },
    "snake_hip_hop_dance": {
        "num_frames": 367,
        "product_name": "Snake Hip Hop Dance",
    },
    "step_hip_hop_dance": {
        "num_frames": 188,
        "product_name": "Step Hip Hop Dance",
    },
    "tut_hip_hop_dance_variation_one": {
        "num_frames": 407,
        "product_name": "Tut Hip Hop Dance Variation One",
    },
    "tut_hip_hop_dance_variation_two": {
        "num_frames": 291,
        "product_name": "Tut Hip Hop Dance Variation Two",
    },
    "wave_hip_hop_dance_variation_one": {
        "num_frames": 404,
        "product_name": "Wave Hip Hop Dance Variation One",
    },
    "wave_hip_hop_dance_variation_two": {
        "num_frames": 384,
        "product_name": "Wave Hip Hop Dance Variation Two",
    },
    "waving_the_arms_hip_hop_dance": {
        "num_frames": 28,
        "product_name": "Waving The Arms Hip Hop Dance",
    },

    # Silly
    "shopping_cart_dance": {
        "num_frames": 383,
        "product_name": "Shopping Cart Dance",
    },
    "silly_dancing_the_cabbage_patch": {
        "num_frames": 92,
        "product_name": "Silly Dancing The Cabbage Patch",
    },
    "silly_dancing_the_twist": {
        "num_frames": 131,
        "product_name": "Silly Dancing The Twist",
    },
    "silly_run": {
        "num_frames": 27,
        "product_name": "Silly Run",
    },
    "turn_to_silly_run_away": {
        "num_frames": 40,
        "product_name": "Turn To Silly Run Away",
    },
    
    # Rumba
    "female_rumba_dancing_loop": {
        "num_frames": 57,
        "product_name": "Female Rumba Dancing -  Loop",
    },
    
    # Breakdance
    "breakdance_arch_freeze_var_3": {
        "num_frames": 148,
        "product_name": "Breakdance Arch Freeze Var 3",
    },
    "breakdance_arch_freeze_var_4": {
        "num_frames": 120,
        "product_name": "Breakdance Arch Freeze Var 4",
    },
    "breakdance_crossleg_freeze_transition_to_stand": {
        "num_frames": 49,
        "product_name": "Breakdance Crossleg Freeze Transition To Stand",
    },
    "breakdance_finishing_combo_var_1": {
        "num_frames": 156,
        "product_name": "Breakdance Finishing Combo Var 1",
    },
    "breakdance_finishing_combo_var_2": {
        "num_frames": 193,
        "product_name": "Breakdance Finishing Combo Var 2",
    },
    "breakdance_finishing_combo_var_3": {
        "num_frames": 154,
        "product_name": "Breakdance Finishing Combo Var 3",
    },
    "breakdance_flair": {
        "num_frames": 24,
        "product_name": "Breakdance Flair",
        "train": False
    },
    "breakdance_flair_end": {
        "num_frames": 60,
        "product_name": "Breakdance Flair End",
        "train": False
    },
    "breakdance_brooklyn_style_uprocking": {
        "num_frames": 117,
        "product_name": "Breakdance Brooklyn Style Uprocking",
        "train": False
    },
    "breakdance_flair_start": {
        "num_frames": 40,
        "product_name": "Breakdance Flair Start",
        "train": False
    },
    "breakdance_freeze_combo": {
        "num_frames": 161,
        "product_name": "Breakdance Freeze Combo",
        "train": False
    },
    "breakdance_ground_footwork_to_crossleg_freeze": {
        "num_frames": 88,
        "product_name": "Breakdance Ground Footwork To Crossleg Freeze",
        "train": False
    },
    "breakdance_ground_footwork_to_hands_on_hips_idle": {
        "num_frames": 231,
        "product_name": "Breakdance Ground Footwork To Hands On Hips Idle",
    },
    "breakdance_ground_footwork_to_stand_idle": {
        "num_frames": 88,
        "product_name": "Breakdance Ground Footwork To Stand Idle",
    },
    "breakdance_ground_footwork_var_1_loop": {
        "num_frames": 76,
        "product_name": "Breakdance Ground Footwork Var 1- Loop",
    },
    "breakdance_ground_footwork_var_2": {
        "num_frames": 39,
        "product_name": "Breakdance Ground Footwork Var 2",
    },
    "breakdance_ground_footwork_var_3": {
        "num_frames": 110,
        "product_name": "Breakdance Ground Footwork Var 3",
    },
    "breakdance_head_spin": {
        "num_frames": 20,
        "product_name": "Breakdance Head Spin",
    },
    "breakdance_headspin_end": {
        "num_frames": 49,
        "product_name": "Breakdance Headspin End",
    },
    "breakdance_headspin_start": {
        "num_frames": 68,
        "product_name": "Breakdance Headspin Start",
    },
    "breakdance_single_handstand_freeze_var_1": {
        "num_frames": 100,
        "product_name": "Breakdance Single Handstand Freeze Var 1",
    },
    "breakdance_single_handstand_freeze_var_2": {
        "num_frames": 147,
        "product_name": "Breakdance Single Handstand Freeze Var 2",
    },
    "breakdance_single_handstand_spin_end": {
        "num_frames": 60,
        "product_name": "Breakdance Single Handstand Spin End",
    },
    "breakdance_single_handstand_spin_start": {
        "num_frames": 84,
        "product_name": "Breakdance Single Handstand Spin Start",
    },
    "breakdance_uprock": {
        "num_frames": 51,
        "product_name": "Breakdance Uprock",
    },
    "breakdance_uprock_footwork_var_2": {
        "num_frames": 128,
        "product_name": "Breakdance Uprock Footwork Var. 2",
    },
    "breakdance_uprock_start": {
        "num_frames": 87,
        "product_name": "Breakdance Uprock Start",
    },
    "breakdance_uprock_transition_to_ground_moves": {
        "num_frames": 109,
        "product_name": "Breakdance Uprock Transition To Ground Moves",
    },
    "breakdance_uprock_var_1": {
        "num_frames": 51,
        "product_name": "Breakdance Uprock Var. 1",
    },
    "ready_to_breakdance_end": {
        "num_frames": 35,
        "product_name": "Ready To Breakdance End",
    },
    "ready_to_breakdance_start": {
        "num_frames": 26,
        "product_name": "Ready To Breakdance Start",
    },
    "start_breakdance_swipes_maneuver": {
        "num_frames": 72,
        "product_name": "Start Breakdance Swipes Maneuver",
    },

    # Capoeira
    "capoeira_cartwheel_move": {
        "num_frames": 70,
        "product_name": "Capoeira Cartwheel Move",
    },
    "capoeira_cartwheel_to_roll_escape": {
        "num_frames": 96,
        "product_name": "Capoeira Cartwheel To Roll Escape",
    },
    "capoeira_flip_escape": {
        "num_frames": 111,
        "product_name": "Capoeira Flip Escape",
    },
    "capoeira_floor_escape": {
        "num_frames": 124,
        "product_name": "Capoeira Floor Escape",
    },
    "capoeira_forward_kick": {
        "num_frames": 72,
        "product_name": "Capoeira Forward Kick",
    },
    "capoeira_ground_spin_kick": {
        "num_frames": 83,
        "product_name": "Capoeira Ground Spin Kick",
    },
    "capoeira_groundwork": {
        "num_frames": 116,
        "product_name": "Capoeira Groundwork",
    },
    "capoeira_high_kick_with_forward_step": {
        "num_frames": 48,
        "product_name": "Capoeira High Kick With Forward Step",
    },
    "capoeira_idle": {
        "num_frames": 82,
        "product_name": "Capoeira Idle",
    },
    "capoeira_kicks": {
        "num_frames": 425,
        "product_name": "Capoeira Kicks",
    },
    "capoeira_retreating_spin_kick": {
        "num_frames": 53,
        "product_name": "Capoeira Retreating Spin Kick",
    },
    "capoeira_side_kick": {
        "num_frames": 33,
        "product_name": "Capoeira Side Kick",
    },
    "capoeira_side_step_flip_escape": {
        "num_frames": 80,
        "product_name": "Capoeira Side Step Flip Escape",
    },
    "capoeira_side_step_spin_kick": {
        "num_frames": 42,
        "product_name": "Capoeira Side Step Spin Kick",
    },
    "capoeira_spin_kick_to_escape": {
        "num_frames": 98,
        "product_name": "Capoeira Spin Kick To Escape",
    },
    "capoeira_step_spin_kick": {
        "num_frames": 57,
        "product_name": "Capoeira Step Spin Kick",
    },
    "capoeira_step_to_cartwheel": {
        "num_frames": 91,
        "product_name": "Capoeira Step To Cartwheel",
    },
    "capoeira_thrust_kick": {
        "num_frames": 33,
        "product_name": "Capoeira Thrust Kick",
    },

    # Swing
    "dance_swing_charleston": {
        "num_frames": 502,
        "product_name": "Dance Swing Charleston",
    },
    "swing_dance_charleston_crazy_legs": {
        "num_frames": 60,
        "product_name": "Swing Dance Charleston \"Crazy Legs\"",
    },
    "swing_dance_charleston_variation_2": {
        "num_frames": 560,
        "product_name": "Swing Dance Charleston Variation 2",
    },
    "swing_dance_shim_sham_variation_1": {
        "num_frames": 593,
        "product_name": "Swing Dance Shim Sham Variation 1",
    },
    "swing_dance_shim_sham_variation_2": {
        "num_frames": 125,
        "product_name": "Swing Dance Shim Sham Variation 2",
        "train": False
    },
    
    # Samba
    "afoxe_samba_reggae_dance": {
        "num_frames": 476,
        "product_name": "Afoxe Samba Reggae Dance",
    },
    "defile_de_samba_parade_variation": {
        "num_frames": 259,
        "product_name": "Defile De Samba Parade Variation",
    },
    "female_fast_samba_enredo_variation_two": {
        "num_frames": 573,
        "product_name": "Female Fast Samba Enredo Variation Two",
    },
    "female_samba_ijexa_break": {
        "num_frames": 657,
        "product_name": "Female Samba Ijexa Break",
        "train": False
    },
    "female_samba_pagode_variation_five_loop": {
        "num_frames": 524,
        "product_name": "Female Samba Pagode Variation Five- Loop",
    },
    "samba_dance_olodum_variation_two": {
        "num_frames": 437,
        "product_name": "Samba Dance Olodum Variation Two",
    },
    "samba_funky_pocoto_variation_1": {
        "num_frames": 669,
        "product_name": "Samba Funky Pocoto Variation 1",
    },
    "samba_gandy_variation_1": {
        "num_frames": 448,
        "product_name": "Samba Gandy Variation 1",
    },

    # House
    "house_dance_variation_3": {
        "num_frames": 513,
        "product_name": "House Dance Variation 3",
    },
    "house_dance_variation_four": {
        "num_frames": 476,
        "product_name": "House Dance Variation Four",
    },
    "house_dance_variation_two": {
        "num_frames": 628,
        "product_name": "House Dance Variation Two",
    },
    
    # African
    "female_african_noodle_dance": {
        "num_frames": 195,
        "product_name": "Female \"African Noodle\" Dance",
    },
    "female_african_rainbow_dance": {
        "num_frames": 176,
        "product_name": "Female \"African Rainbow\" Dance",
    },
    
    # Soul
    "basic_northern_soul_step": {
        "num_frames": 386,
        "product_name": "Basic Northern Soul Step",
    },
    "northern_soul_dance_spin": {
        "num_frames": 97,
        "product_name": "Northern Soul Dance Spin",
    },
    "northern_soul_dance_spin_and_floor_work": {
        "num_frames": 255,
        "product_name": "Northern Soul Dance Spin And Floor Work",
    },
    "northern_soul_dance_spin_dip_and_splits": {
        "num_frames": 212,
        "product_name": "Northern Soul Dance Spin Dip And Splits",
    },
    "northern_soul_dance_spin_on_floor": {
        "num_frames": 180,
        "product_name": "Northern Soul Dance Spin On Floor",
    },

    # Salsa
    "female_salsa_dancing": {
        "num_frames": 108,
        "product_name": "Female Salsa Dancing",
        "train": False
    },
    "male_partner_salsa_variation_five": {
        "num_frames": 675,
        "product_name": "Male Partner Salsa Variation Five",
        "train": False
    },
    "male_partner_salsa_variation_one": {
        "num_frames": 456,
        "product_name": "Male Partner Salsa Variation One",
    },
    "male_partner_salsa_variation_two": {
        "num_frames": 452,
        "product_name": "Male Partner Salsa Variation Two",
    },
    "male_salsa_variation_eight": {
        "num_frames": 518,
        "product_name": "Male Salsa Variation Eight",
    },
    "salsa_dance_variation_five": {
        "num_frames": 288,
        "product_name": "Salsa Dance Variation Five",
    },
    "salsa_dancing_double_twirl": {
        "num_frames": 63,
        "product_name": "Salsa Dancing Double Twirl",
    },
    "salsa_dancing_side_to_side": {
        "num_frames": 55,
        "product_name": "Salsa Dancing Side To Side",
    },
    "salsa_dancing_twirl_and_clap": {
        "num_frames": 55,
        "product_name": "Salsa Dancing Twirl And Clap",
    },

    # Thriller
    "thriller_dance_part_1": {
        "num_frames": 716,
        "product_name": "Thriller Dance Part 1",
        "train": False
    },
    "thriller_dance_part_2": {
        "num_frames": 452,
        "product_name": "Thriller Dance Part 2",
    },
    "thriller_dance_part_3": {
        "num_frames": 614,
        "product_name": "Thriller Dance Part 3",
    },
    "thriller_dance_part_4": {
        "num_frames": 890,
        "product_name": "Thriller Dance Part 4",
    },

    # Jazz
    "female_jazz_dancing_around_the_world": {
        "num_frames": 66,
        "product_name": "Female Jazz Dancing 'Around The World'",
    },
    "female_jazz_dancing_around_the_world_loop": {
        "num_frames": 55,
        "product_name": "Female Jazz Dancing 'Around The World'- Loop",
    },
    "female_jazz_dancing_rockette_kick": {
        "num_frames": 131,
        "product_name": "Female Jazz Dancing 'Rockette Kick'",
    },
    "female_jazz_square_dancing": {
        "num_frames": 49,
        "product_name": "Female Jazz Square Dancing",
    },
     
    # Other
    "baseball_bunt_swing": {
        "num_frames": 48,
        "product_name": "Baseball Bunt Swing",
    },
    "baseball_swinging_strike": {
        "num_frames": 167,
        "product_name": "Baseball Swinging Strike",
    },
    "bellydance_variation_2": {
        "num_frames": 609,
        "product_name": "Bellydance Variation 2",
    },
    "dancing_the_twerk": {
        "num_frames": 365,
        "product_name": "Dancing The Twerk",
    },
    "doing_the_can_can": {
        "num_frames": 88,
        "product_name": "Doing The Can Can",
    },
    "doing_the_chicken_dance": {
        "num_frames": 115,
        "product_name": "Doing The Chicken Dance",
    },
    "doing_the_hokey_pokey": {
        "num_frames": 280,
        "product_name": "Doing The Hokey Pokey",
    },
    "doing_the_shuffling_dance": {
        "num_frames": 180,
        "product_name": "Doing The Shuffling Dance",
    },
    "doing_the_twist_dance": {
        "num_frames": 284,
        "product_name": "Doing The Twist Dance",
    },
    "doing_the_ymca_dance": {
        "num_frames": 108,
        "product_name": "Doing The Ymca Dance",
    },
    "male_walking_with_shopping_bag": {
        "num_frames": 31,
        "product_name": "Male Walking With Shopping Bag",
    },
    "russian_kettlebell_swing": {
        "num_frames": 48,
        "product_name": "Russian Kettlebell Swing",
    },
    "the_popular_kpop_dance": {
        "num_frames": 297,
        "product_name": "The Popular K-Pop Dance",
    },
    "walking_backwards_sliding_feet": {
        "num_frames": 25,
        "product_name": "Walking Backwards Sliding Feet",
    },
    "watering_with_a_hose": {
        "num_frames": 135,
        "product_name": "Watering With A Hose",
    },
}

################################################################################
################################################################################
################################################################################
################################################################################

animations_by_identity = {
    "adam": [
        "bboy_hip_hop_variation_one", # 53
        "bboy_uprock_start", # 44
        "bboy_pose_to_standing_idle", # 56
        "bboy_uprock_indian_step", # 56
        "booty_step_hip_hop_dance", # 118
        "female_hip_hop_arm_wave_dancing", # 125
        "male_salsa_variation_eight", # 518
        "capoeira_high_kick_with_forward_step", # 48
        "hip_hop_dancing_bboy_variation_one", # 331
    ],
    "alex": [
        "female_hip_hop_body_wave_dancing", # 170
        "female_hip_hop_kick_step_dancing", # 152
        "female_hip_hop_rib_pops_dancing", # 81
        "female_hip_hop_running_man_dancing", # 114
        "salsa_dance_variation_five", # 288
        "salsa_dancing_double_twirl", # 63
        "salsa_dancing_side_to_side", # 55
        "salsa_dancing_twirl_and_clap", # 55
        "capoeira_idle", # 82
        "breakdance_finishing_combo_var_2", # 193
    ],
    "alien": [
        # TEST
        "female_hip_hop_slide_step_dancing", # 108
        "hip_hop_dance_moonwalk", # 414
        "thriller_dance_part_1", # 716
    ],
    "astra": [
        "hip_hop_dancing_bboy_variation_one", #331
        "hip_hop_dancing_really_twirl", # 376
        "thriller_dance_part_2", # 452
        "capoeira_retreating_spin_kick", # 53
    ],
    "brian": [
        "hip_hop_dancing_shimmy", # 288
        "hip_hop_dancing_side_to_side", # 328
        "thriller_dance_part_3", # 614
        "capoeira_side_kick", # 33
    ],
    "bryce": [
        "hip_hop_just_listening_dancing_variation", # 446
        "thriller_dance_part_4", # 890
        "capoeira_side_step_flip_escape", # 80
    ],
    "chad": [
        # VAL
        "hip_hop_quake_variation_1", # 379
        "hip_hop_robot_dance_variation_two", # 368
    ],
    "crypto": [
        "hip_hop_runningman_dance", # 147
        "locking_hip_hop_dance_variation_two", # 408
        "female_jazz_dancing_around_the_world", # 66
        "female_jazz_dancing_rockette_kick", # 131
        "female_jazz_square_dancing", # 49
        "baseball_bunt_swing", # 48
        "baseball_swinging_strike", # 167
        "capoeira_side_step_spin_kick", # 42
        "thriller_dance_part_4", # 890
    ],
    "david": [
        "robot_hip_hop_dance_variation_one", # 371
        "slide_hip_hop_dance_variation_one", # 416
        "female_jazz_dancing_around_the_world_loop", # 55
        "dancing_the_twerk", # 365
        "capoeira_spin_kick_to_escape", # 98
        "baseball_swinging_strike", # 167
    ],
    "douglas": [
        "slide_hip_hop_walk", # 24
        "snake_hip_hop_dance", # 367
        "step_hip_hop_dance", # 188
        "bellydance_variation_2", # 609
        "capoeira_step_spin_kick", # 57
        "female_jazz_dancing_around_the_world_loop", # 55
    ],
    "elizabeth": [
        "tut_hip_hop_dance_variation_one", # 407
        "tut_hip_hop_dance_variation_two", # 291
        "doing_the_can_can", # 88
        "doing_the_chicken_dance", # 115
        "doing_the_hokey_pokey", # 208
        "capoeira_step_to_cartwheel", # 91
        "female_jazz_dancing_around_the_world", # 66
    ],
    "eve": [
        "wave_hip_hop_dance_variation_one", # 404
        "doing_the_shuffling_dance", # 180
        "doing_the_twist_dance", # 284
        "doing_the_ymca_dance", # 108
        "capoeira_thrust_kick", # 33
        "female_jazz_dancing_rockette_kick", # 131
    ],
    # "exo": [
    #     "wave_hip_hop_dance_variation_two", # 384
    #     "waving_the_arms_hip_hop_dance", # 28
    #     "male_walking_with_shopping_bag", # 31
    #     "russian_kettlebell_swing", # 48
    #     "the_popular_kpop_dance", # 297
    #     "walking_backwards_sliding_feet", # 25
    #     "watering_with_a_hose", # 135
    #     "dance_swing_charleston", # 502
    #     "female_jazz_square_dancing", # 49
    #     "baseball_bunt_swing", # 48
    # ],
    "james": [
        "shopping_cart_dance", # 383
        "silly_dancing_the_cabbage_patch", # 92
        "bboy_hip_hop_variation_one", # 53
        "bboy_uprock_start", # 44
        "bboy_pose_to_standing_idle", # 56
        "bboy_uprock_indian_step", # 56
        "booty_step_hip_hop_dance", # 118
        "female_hip_hop_arm_wave_dancing", # 125
        "swing_dance_charleston_crazy_legs", # 60
        "thriller_dance_part_2", # 452
    ],
    "jennifer": [
        "silly_dancing_the_twist", # 131
        "silly_run", # 27
        "turn_to_silly_run_away", # 40
        "female_rumba_dancing_loop", # 57
        "breakdance_arch_freeze_var_3", # 148
        "breakdance_arch_freeze_var_4", # 120
        "female_hip_hop_body_wave_dancing", # 170
        "female_hip_hop_kick_step_dancing", # 152
        "female_hip_hop_rib_pops_dancing", # 81
        "swing_dance_charleston_variation_2", # 560
    ],
    "jody": [
        "breakdance_crossleg_freeze_transition_to_stand", # 49
        "breakdance_finishing_combo_var_1", # 156
        "breakdance_finishing_combo_var_3", # 154
        "female_hip_hop_running_man_dancing", # 114
        "swing_dance_shim_sham_variation_1", # 593
        "bellydance_variation_2", # 609
    ],
    "joe": [
        # TEST
        "breakdance_flair", # 24
        "breakdance_flair_end", # 60
        "breakdance_brooklyn_style_uprocking", # 117
        "breakdance_flair_start", # 40
        "breakdance_freeze_combo", # 161
        "breakdance_ground_footwork_to_crossleg_freeze", # 88
    ],
    "josh": [
        "breakdance_ground_footwork_to_hands_on_hips_idle", # 231
        "breakdance_ground_footwork_to_stand_idle", # 88
        "breakdance_ground_footwork_var_1_loop", # 76
        "breakdance_ground_footwork_var_2", # 39
        "breakdance_ground_footwork_var_3", # 110
        "breakdance_head_spin", # 20
        "breakdance_headspin_end", # 49
        "hip_hop_dancing_really_twirl", # 376
        "hip_hop_dancing_shimmy", # 288
        "afoxe_samba_reggae_dance", # 476
    ],
    "kate": [
        "breakdance_headspin_start", # 68
        "breakdance_single_handstand_freeze_var_1", # 100
        "breakdance_single_handstand_freeze_var_2", # 147
        "breakdance_single_handstand_spin_end", # 60
        "breakdance_single_handstand_spin_start", # 84
        "breakdance_uprock", # 51
        "hip_hop_dancing_side_to_side", # 328
        "hip_hop_just_listening_dancing_variation", # 446
        "defile_de_samba_parade_variation", # 259
    ],
    "leonard": [
        "breakdance_uprock_footwork_var_2", # 128
        "breakdance_uprock_start", # 87
        "breakdance_uprock_transition_to_ground_moves", # 109
        "breakdance_uprock_var_1", # 51
        "ready_to_breakdance_end", # 35
        "ready_to_breakdance_start", # 26
        "start_breakdance_swipes_maneuver", # 72
        "hip_hop_runningman_dance", # 147
        "locking_hip_hop_dance_variation_two", # 408
        "female_fast_samba_enredo_variation_two", # 573
    ],
    "lewis": [
        "capoeira_cartwheel_move", # 70
        "capoeira_cartwheel_to_roll_escape", # 96
        "capoeira_flip_escape", # 111
        "capoeira_floor_escape", # 124
        "capoeira_forward_kick", # 72
        "capoeira_ground_spin_kick", # 83
        "capoeira_groundwork", # 116
        "robot_hip_hop_dance_variation_one", # 371
        "slide_hip_hop_dance_variation_one", # 416
        "female_samba_pagode_variation_five_loop", # 524
    ],
    # "liam": [
    #     "capoeira_high_kick_with_forward_step", # 48
    #     "capoeira_idle", # 82
    #     "capoeira_kicks", # 425
    #     "slide_hip_hop_walk", # 24
    #     "snake_hip_hop_dance", # 367
    #     "step_hip_hop_dance", # 188
    #     "samba_dance_olodum_variation_two", # 437
    #     "dancing_the_twerk", # 365
    # ],
    "louise": [
        "capoeira_retreating_spin_kick", # 53
        "capoeira_side_kick", # 33
        "capoeira_side_step_flip_escape", # 80
        "capoeira_side_step_spin_kick", # 42
        "capoeira_spin_kick_to_escape", # 98
        "capoeira_step_spin_kick", # 57
        "capoeira_step_to_cartwheel", # 91
        "capoeira_thrust_kick", # 33
        "tut_hip_hop_dance_variation_one", # 407
        "wave_hip_hop_dance_variation_two", # 384
    ],
    # "malcom": [
    #     "dance_swing_charleston", # 502
    #     "swing_dance_charleston_crazy_legs", # 60
    #     "tut_hip_hop_dance_variation_two", # 291
    #     "wave_hip_hop_dance_variation_one", # 384
    #     "male_partner_salsa_variation_two", # 452
    # ],
    "mannequin": [
        # VAL
        "swing_dance_shim_sham_variation_2", # 125
    ],
    # "markerman": [
    #     "swing_dance_shim_sham_variation_1", # 593
    #     "waving_the_arms_hip_hop_dance", # 28
    #     "shopping_cart_dance", # 383
    #     "silly_dancing_the_cabbage_patch", # 92
    #     "samba_funky_pocoto_variation_1", # 669
    #     "doing_the_can_can", # 88
    # ],
    "martha": [
        "swing_dance_charleston_variation_2", # 560
        "silly_dancing_the_twist", # 131
        "silly_run", # 27
        "turn_to_silly_run_away", # 40
        "female_rumba_dancing_loop", # 57
        "breakdance_arch_freeze_var_3", # 148
        "doing_the_chicken_dance", # 115
        "doing_the_hokey_pokey", # 280
    ],
    "megan": [
        "afoxe_samba_reggae_dance", # 476
        "defile_de_samba_parade_variation", # 259
        "breakdance_arch_freeze_var_4", # 120
        "breakdance_crossleg_freeze_transition_to_stand", # 49
        "breakdance_finishing_combo_var_1", # 156
        "breakdance_finishing_combo_var_2", # 193
        "doing_the_shuffling_dance", # 180
    ],
    "ninja": [
        "female_fast_samba_enredo_variation_two", # 573
        "breakdance_finishing_combo_var_3", # 154
        "breakdance_ground_footwork_to_hands_on_hips_idle", # 231
        "breakdance_ground_footwork_to_stand_idle", # 88
        "breakdance_ground_footwork_var_1_loop", # 76
        "doing_the_twist_dance", # 284
    ],
    "olivia": [
        # TEST
        "female_samba_ijexa_break", # 657
    ],
    "pete": [
        "female_samba_pagode_variation_five_loop", # 524
        "breakdance_ground_footwork_var_2", # 39
        "breakdance_ground_footwork_var_3", # 110
        "breakdance_head_spin", # 20
        "breakdance_headspin_end", # 49
        "capoeira_kicks", # 425
    ],
    "racer": [
        "samba_dance_olodum_variation_two", # 437
        "breakdance_headspin_start", # 68
        "breakdance_single_handstand_freeze_var_1", # 100
        "male_partner_salsa_variation_one", # 456
        "doing_the_ymca_dance", # 108
    ],
    "regina": [
        "samba_funky_pocoto_variation_1", # 669
        "breakdance_single_handstand_freeze_var_2", # 147
        "breakdance_single_handstand_spin_end", # 60
        "northern_soul_dance_spin", # 97
        "northern_soul_dance_spin_and_floor_work", # 255
        "male_salsa_variation_eight", # 518
    ],
    # "remy": [
    #     "samba_gandy_variation_1", # 448
    #     "breakdance_single_handstand_spin_start", # 84
    #     "breakdance_uprock", # 51
    #     "breakdance_uprock_footwork_var_2", # 128
    #     "basic_northern_soul_step", # 386
    #     "salsa_dance_variation_five", # 288
    #     "salsa_dancing_double_twirl", # 63
    # ],
    "romero": [
        "house_dance_variation_3", # 513
        "breakdance_uprock_start", # 87
        "breakdance_uprock_transition_to_ground_moves", # 109
        "samba_gandy_variation_1", # 448
        "salsa_dancing_side_to_side", # 55
        "salsa_dancing_twirl_and_clap", # 55
    ],
    "roth": [
        "house_dance_variation_four", # 476
        "breakdance_uprock_var_1", # 51
        "ready_to_breakdance_end", # 35
        "female_african_noodle_dance", # 195
        "female_african_rainbow_dance", # 176
        "northern_soul_dance_spin_dip_and_splits", # 212
        "male_walking_with_shopping_bag", # 31
    ],
    "shae": [
        "house_dance_variation_two", # 628
        "ready_to_breakdance_start", # 26
        "start_breakdance_swipes_maneuver", # 72
        "capoeira_cartwheel_move", # 70
        "thriller_dance_part_3", # 614
    ],
    "shannon": [
        "female_african_noodle_dance", # 195
        "female_african_rainbow_dance", # 176
        "basic_northern_soul_step", # 386 
        "capoeira_cartwheel_to_roll_escape", # 96
        "watering_with_a_hose", # 135
        "house_dance_variation_two", # 628
    ],
    "sophie": [
        # TEST
        "female_salsa_dancing", # 108
        "male_partner_salsa_variation_five", # 675
    ],
    "steve": [
        "northern_soul_dance_spin", # 97
        "northern_soul_dance_spin_and_floor_work", # 255
        "northern_soul_dance_spin_dip_and_splits", # 212
        "northern_soul_dance_spin_on_floor", # 180
        "capoeira_flip_escape", # 111
        "russian_kettlebell_swing", # 48
        "the_popular_kpop_dance", # 297
        "walking_backwards_sliding_feet", # 25
    ],
    "suzie": [
        "male_partner_salsa_variation_one", # 456
        "capoeira_floor_escape", # 124
        "capoeira_forward_kick", # 72
        "house_dance_variation_3", # 513
    ],
    "swatguy": [
        "male_partner_salsa_variation_two", # 452
        "capoeira_ground_spin_kick", # 83
        "capoeira_groundwork", # 116
        "house_dance_variation_four", # 476
        "northern_soul_dance_spin_on_floor", # 180
    ],
}


def compute_dataset_statistics(animation_info):
    
    num_animations = len(animation_info)

    # Number of unique frames
    num_unique_frames = 0
    num_unique_frames_train = 0
    num_unique_frames_test = 0
    for anim_data in animation_info.values():
        num_unique_frames += anim_data['num_frames']
        if 'train' not in anim_data:
            num_unique_frames_train += anim_data['num_frames']
        else:
            num_unique_frames_test += anim_data['num_frames']

    # Dataset
    num_frames_by_identity = {k: 0 for k in animations_by_identity.keys()}
    num_animation_appearances = {k: 0 for k in animation_info.keys()}

    for identity_name, identity_anims in animations_by_identity.items():
        for anim_name in identity_anims:
            
            num_animation_appearances[anim_name] += 1
            
            n = animation_info[anim_name]['num_frames']
            num_frames_by_identity[identity_name] += n


    return {
        "num_animations": num_animations,
        "num_unique_frames": num_unique_frames,
        "num_unique_frames_train": num_unique_frames_train,
        "num_unique_frames_test": num_unique_frames_test,
        "num_frames_by_identity": num_frames_by_identity,
        "num_animation_appearances": num_animation_appearances
    }


if __name__ == "__main__":

    # all_anims = []

    # for style, anims in animations_by_style.items():
        
    #     print()
    #     print()

    #     for anim in sorted(anims):
            
    #         print(anim)
            
    #         all_anims.append(anim)


    # for anim in sorted(all_anims):
    #     print(anim)

    stats = compute_dataset_statistics(animation_info)

    print("Total num UNIQUE frames          ", stats['num_unique_frames'])
    print("Total num UNIQUE frames train    ", stats['num_unique_frames_train'])
    print("Total num UNIQUE frames test/val ", stats['num_unique_frames_test'])
    print()

    total_num_frames = 0
    total_num_train_frames = 0
    total_num_test_frames = 0

    for identity, num_frames in stats['num_frames_by_identity'].items():
        print(identity, num_frames)
        total_num_frames += num_frames
        
        if identity in test_identities or identity in val_identities:
            total_num_test_frames += num_frames
        else:
            total_num_train_frames += num_frames


    print()
    print(f"Total num frames {total_num_frames} (train {total_num_train_frames} - test/val {total_num_test_frames})")
    print()

    for anim, appearances in stats['num_animation_appearances'].items():
        # print(anim, appearances)

        if "train" in animation_info[anim]:
            assert appearances == 1, anim
        else:
            assert appearances <= 2, anim
        
    potential_num_frames = stats['num_unique_frames_train'] * 2 + stats['num_unique_frames_test']
    
    # assert total_num_frames == potential_num_frames, f"{total_num_frames} vs {potential_num_frames}"

