
## from infos.py from official d4rl github repo
REF_MIN_SCORE = {
    'halfcheetah' : -280.178953,
    'walker2d' : 1.629008,
    'hopper' : -20.272305,
    'ant' : -325.6,
    'antmaze' : 0.0,
}
 
REF_MAX_SCORE = {
    'halfcheetah' : 12135.0,
    'walker2d' : 4592.3,
    'hopper' : 3234.3,
    'ant' : 3879.7,
    'antmaze' : 700,
}

## calculated from d4rl datasets
D4RL_DATASET_STATS = {
        'halfcheetah-medium-v2': {
                'state_mean':[-0.06845773756504059, 0.016414547339081764, -0.18354906141757965,
                              -0.2762460708618164, -0.34061527252197266, -0.09339715540409088,
                              -0.21321271359920502, -0.0877423882484436, 5.173007488250732,
                              -0.04275195300579071, -0.036108363419771194, 0.14053793251514435,
                              0.060498327016830444, 0.09550975263118744, 0.06739100068807602,
                              0.005627387668937445, 0.013382787816226482
                ],
                'state_std':[0.07472999393939972, 0.3023499846458435, 0.30207309126853943,
                             0.34417077898979187, 0.17619241774082184, 0.507205605506897,
                             0.2567007839679718, 0.3294812738895416, 1.2574149370193481,
                             0.7600541710853577, 1.9800915718078613, 6.565362453460693,
                             7.466367721557617, 4.472222805023193, 10.566964149475098,
                             5.671932697296143, 7.4982590675354
                ]
            },
        'halfcheetah-medium-replay-v2': {
                'state_mean':[-0.12880703806877136, 0.3738119602203369, -0.14995987713336945,
                              -0.23479078710079193, -0.2841278612613678, -0.13096535205841064,
                              -0.20157982409000397, -0.06517726927995682, 3.4768247604370117,
                              -0.02785065770149231, -0.015035249292850494, 0.07697279006242752,
                              0.01266712136566639, 0.027325302362442017, 0.02316424623131752,
                              0.010438721626996994, -0.015839405357837677
                ],
                'state_std':[0.17019015550613403, 1.284424901008606, 0.33442774415016174,
                             0.3672759234905243, 0.26092398166656494, 0.4784106910228729,
                             0.3181420564651489, 0.33552637696266174, 2.0931615829467773,
                             0.8037433624267578, 1.9044333696365356, 6.573209762573242,
                             7.572863578796387, 5.069749355316162, 9.10555362701416,
                             6.085654258728027, 7.25300407409668
                ]
            },
        'halfcheetah-medium-expert-v2': {
                'state_mean':[-0.05667462572455406, 0.024369969964027405, -0.061670560389757156,
                              -0.22351515293121338, -0.2675151228904724, -0.07545716315507889,
                              -0.05809682980179787, -0.027675075456500053, 8.110626220703125,
                              -0.06136331334710121, -0.17986927926540375, 0.25175222754478455,
                              0.24186332523822784, 0.2519369423389435, 0.5879552960395813,
                              -0.24090635776519775, -0.030184272676706314
                ],
                'state_std':[0.06103534251451492, 0.36054104566574097, 0.45544400811195374,
                             0.38476887345314026, 0.2218363732099533, 0.5667523741722107,
                             0.3196682929992676, 0.2852923572063446, 3.443821907043457,
                             0.6728139519691467, 1.8616976737976074, 9.575807571411133,
                             10.029894828796387, 5.903450012207031, 12.128185272216797,
                             6.4811787605285645, 6.378620147705078
                ]
            },
        'walker2d-medium-v2': {
                'state_mean':[1.218966007232666, 0.14163373410701752, -0.03704913705587387,
                              -0.13814310729503632, 0.5138224363327026, -0.04719110205769539,
                              -0.47288352251052856, 0.042254164814949036, 2.3948874473571777,
                              -0.03143199160695076, 0.04466355964541435, -0.023907244205474854,
                              -0.1013401448726654, 0.09090937674045563, -0.004192637279629707,
                              -0.12120571732521057, -0.5497063994407654
                ],
                'state_std':[0.12311358004808426, 0.3241879940032959, 0.11456084251403809,
                             0.2623065710067749, 0.5640279054641724, 0.2271878570318222,
                             0.3837319612503052, 0.7373676896095276, 1.2387926578521729,
                             0.798020601272583, 1.5664079189300537, 1.8092705011367798,
                             3.025604248046875, 4.062486171722412, 1.4586567878723145,
                             3.7445690631866455, 5.5851287841796875
                ]
            },
        'walker2d-medium-replay-v2': {
                'state_mean':[1.209364652633667, 0.13264022767543793, -0.14371201395988464,
                              -0.2046516090631485, 0.5577612519264221, -0.03231537342071533,
                              -0.2784661054611206, 0.19130706787109375, 1.4701707363128662,
                              -0.12504704296588898, 0.0564953051507473, -0.09991033375263214,
                              -0.340340256690979, 0.03546293452382088, -0.08934258669614792,
                              -0.2992438077926636, -0.5984178185462952
                ],
                'state_std':[0.11929835379123688, 0.3562574088573456, 0.25852200388908386,
                             0.42075422406196594, 0.5202291011810303, 0.15685082972049713,
                             0.36770978569984436, 0.7161387801170349, 1.3763766288757324,
                             0.8632221817970276, 2.6364643573760986, 3.0134117603302,
                             3.720684051513672, 4.867283821105957, 2.6681625843048096,
                             3.845186948776245, 5.4768385887146
                ]
            },
        'walker2d-medium-expert-v2': {
                'state_mean':[1.2294334173202515, 0.16869689524173737, -0.07089081406593323,
                              -0.16197483241558075, 0.37101927399635315, -0.012209027074277401,
                              -0.42461398243904114, 0.18986578285694122, 3.162475109100342,
                              -0.018092676997184753, 0.03496946766972542, -0.013921679928898811,
                              -0.05937029421329498, -0.19549426436424255, -0.0019200450042262673,
                              -0.062483321875333786, -0.27366524934768677
                ],
                'state_std':[0.09932824969291687, 0.25981399416923523, 0.15062759816646576,
                             0.24249176681041718, 0.6758718490600586, 0.1650741547346115,
                             0.38140663504600525, 0.6962361335754395, 1.3501490354537964,
                             0.7641991376876831, 1.534574270248413, 2.1785972118377686,
                             3.276582717895508, 4.766193866729736, 1.1716983318328857,
                             4.039782524108887, 5.891613960266113
                ]
            },
        'hopper-medium-v2': {
                'state_mean':[1.311279058456421, -0.08469521254301071, -0.5382719039916992,
                              -0.07201576232910156, 0.04932365566492081, 2.1066856384277344,
                              -0.15017354488372803, 0.008783451281487942, -0.2848185896873474,
                              -0.18540096282958984, -0.28461286425590515
                ],
                'state_std':[0.17790751159191132, 0.05444620922207832, 0.21297138929367065,
                             0.14530418813228607, 0.6124444007873535, 0.8517446517944336,
                             1.4515252113342285, 0.6751695871353149, 1.5362390279769897,
                             1.616074562072754, 5.607253551483154
                ]
            },
        'hopper-medium-replay-v2': {
                'state_mean':[1.2305138111114502, -0.04371410980820656, -0.44542956352233887,
                              -0.09370097517967224, 0.09094487875699997, 1.3694725036621094,
                              -0.19992674887180328, -0.022861352190375328, -0.5287045240402222,
                              -0.14465883374214172, -0.19652697443962097
                ],
                'state_std':[0.1756512075662613, 0.0636928603053093, 0.3438323438167572,
                             0.19566889107227325, 0.5547984838485718, 1.051029920578003,
                             1.158307671546936, 0.7963128685951233, 1.4802359342575073,
                             1.6540331840515137, 5.108601093292236
                ]
            },
        'hopper-medium-expert-v2': {
                'state_mean':[1.3293815851211548, -0.09836531430482864, -0.5444297790527344,
                              -0.10201650857925415, 0.02277466468513012, 2.3577215671539307,
                              -0.06349576264619827, -0.00374026270583272, -0.1766270101070404,
                              -0.11862941086292267, -0.12097819894552231
                ],
                'state_std':[0.17012375593185425, 0.05159067362546921, 0.18141433596611023,
                             0.16430604457855225, 0.6023368239402771, 0.7737284898757935,
                             1.4986555576324463, 0.7483318448066711, 1.7953159809112549,
                             2.0530025959014893, 5.725032806396484
                ]
            },
        'ant-medium-v2': {
            'state_mean':[5.8879435e-01,  8.7174654e-01, -7.1400687e-02, -5.3334899e-02,
                            4.0101001e-01, -1.0269742e-01,  5.4115975e-01,  5.2002072e-01,
                            -5.4039919e-01,  3.9520892e-03, -7.1784544e-01, -5.0022161e-01,
                            6.0683137e-01,  3.6226776e+00, -1.1381905e-01,  2.3744591e-03,
                            -2.3013927e-02, -1.9157669e-02,  1.3230776e-02,  1.1581083e-02,
                            1.7401379e-02, -3.9027195e-04, -1.8636726e-02,  1.3028152e-02,
                            -2.0028191e-02,  9.2588821e-03,  2.6023872e-02,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,        
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                            0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
            'state_std':[1.2383466e-01, 1.4422522e-01, 1.2160219e-01, 1.0867277e-01,
                        1.5538533e-01, 4.1339096e-01, 9.5701441e-02, 7.8999139e-02,
                        1.0126759e-01, 4.3590635e-01, 2.1259798e-01, 1.1442706e-01,
                        1.7129923e-01, 1.2775923e+00, 9.0014160e-01, 1.1273303e+00,
                        1.3589811e+00, 1.0598884e+00, 1.0320269e+00, 5.7127028e+00,
                        8.1173664e-01, 5.4215431e-01, 6.1074728e-01, 5.9005790e+00,
                        2.7778540e+00, 9.1609848e-01, 2.0859509e+00, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06]
        },
        'ant-medium-replay-v2': {
            'state_mean':[5.11797667e-01,  6.25735641e-01, -9.19744298e-02,  2.89481524e-02,
                        3.83356541e-01, -1.79010451e-01,  6.38916492e-01,  3.76463264e-01,
                        -7.09240198e-01,  1.00261301e-01, -8.47647130e-01, -2.28275016e-01,
                        7.43384123e-01,  1.26649547e+00, -1.48257434e-01,  2.18457490e-06,
                        -1.49968825e-02, -2.19108956e-03,  3.91414128e-02, -6.59901707e-05,
                        2.28942465e-02,  1.51484378e-03, -2.61836853e-02,  1.11463768e-02,
                        -2.65279412e-02,  1.59643719e-03,  2.40335930e-02,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            'state_std':[1.4633577e-01, 4.1725650e-01, 2.6143208e-01, 3.0398563e-01,
                        3.4250414e-01, 4.1728887e-01, 2.4019210e-01, 3.3658993e-01,
                        2.9806718e-01, 4.5031688e-01, 2.9821286e-01, 4.5226517e-01,
                        2.8720662e-01, 1.5094841e+00, 6.7063022e-01, 7.5184345e-01,
                        1.0571648e+00, 8.4482086e-01, 1.0695845e+00, 4.0341158e+00,
                        1.3410360e+00, 1.3760650e+00, 1.2430267e+00, 4.3622222e+00,
                        2.3340695e+00, 1.2857327e+00, 2.3173475e+00, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06, 1.0000000e-06,
                        1.0000000e-06, 1.0000000e-06, 1.0000000e-06]
        },
        'antmaze-umaze-v2': {
            'state_mean': [3.99919438e+00,7.04566479e+00,4.90209639e-01,6.31213307e-01,
            -2.03014761e-02,1.24627471e-01,-1.79573417e-01,3.65527049e-02,
            7.08520234e-01,6.17269650e-02,-7.75078058e-01,-1.12521425e-01,
            -6.98012888e-01,3.10450569e-02,7.37622738e-01,4.22175564e-02,
            1.17438957e-01,-6.19870902e-04,-5.35303354e-03,1.94301596e-03,
            -1.36638097e-02,-2.45976984e-03,1.65986214e-02,-2.38764519e-03,
            -1.56380888e-02,1.48862472e-03,-1.64170023e-02,6.59266952e-03,
            1.54044479e-02],
            'state_std': [2.7842069,3.1524637,0.15707897,0.4107535,0.3299878,0.34344134,
            0.3979672,0.45365378,0.27704316,0.42626643,0.302641,0.4286146,
            0.2684806,0.4342764,0.28686318,0.81600064,0.74595106,0.69215363,
            1.0526766,1.0722063,1.084649,2.4674027,1.8194557,2.7185004,
            1.9193915,2.7246525,1.7709186,2.4674568,1.6988297],
            },
            
        'antmaze-umaze-diverse-v2': {
            'state_mean': [3.03271222e+00,8.30902195e+00,4.66011286e-01,5.77190638e-01,
            1.61201954e-02,1.42301232e-01,-1.56136960e-01,6.08957335e-02,
            7.00020730e-01,7.91066065e-02,-8.04380536e-01,-1.41956538e-01,
            -6.91604018e-01,4.83990125e-02,7.25090742e-01,-3.31155546e-02,
            4.72433269e-02,6.11345749e-04,-4.68651677e-04,3.32747540e-03,
            -8.12269468e-03,-4.70714178e-03,1.20742591e-02,2.94493465e-03,
            -1.19334953e-02,1.04567816e-03,-1.47810867e-02,4.90594329e-03,
            1.41088255e-02],
            'state_std': [2.3728664,1.6758876,0.16024312,0.44831368,0.36744747,0.37735796,
            0.37907827,0.4571016,0.2725728,0.43000388,0.3134053,0.4271536,
            0.2681549,0.4363314,0.2832151,0.6943002,0.68571335,0.65786135,
            0.96929276,1.0120589,1.0464511,2.3568401,1.6926415,2.5428424,
            1.8539752,2.6219027,1.5936253,2.316399,1.5101156],
            },
            
        'antmaze-medium-diverse-v2': {
            'state_mean': [1.20002480e+01,1.29046755e+01,4.84451652e-01,5.40328562e-01,
            1.94956027e-02,3.17021087e-02,9.08246711e-02,-2.73967460e-02,
            6.83756828e-01,1.07427396e-01,-7.04916894e-01,-4.88132983e-02,
            -7.85228133e-01,1.42699871e-02,6.90363050e-01,2.81006694e-02,
            3.33977230e-02,3.17373284e-04,2.22907588e-03,1.91757154e-05,
            2.23926944e-03,-3.55758006e-04,1.18721658e-02,1.25071674e-04,
            -1.19093312e-02,-3.66920303e-03,-1.24379909e-02,4.42140922e-03,
            1.25643201e-02],
            'state_std': [7.147566,6.1185136,0.15346959,0.40846276,0.34429362,0.3436737,
            0.54289216,0.43955532,0.26366585,0.43549758,0.27824324,0.44196016,
            0.30912885,0.4481787,0.26372364,0.67349714,0.69270283,0.64122987,
            0.980725,1.0179923,1.0749477,2.364969,1.7413058,2.5638735,
            1.5949968,2.5273018,1.7363882,2.3930802,1.6383512],
            },
            
        'antmaze-medium-play-v2': {
            'state_mean': [1.21532774e+01,1.30323887e+01,4.81982976e-01,5.17669022e-01,
            2.56814621e-02,3.68972421e-02,6.16297200e-02,-3.09576355e-02,
            6.85755849e-01,1.16558217e-01,-7.04945743e-01,-4.78746369e-02,
            -7.78221190e-01,1.31647615e-02,6.98675990e-01,2.16756985e-02,
            3.07137128e-02,7.11303030e-04,2.89444206e-03,1.24167022e-03,
            2.16782582e-03,-3.26603564e-04,1.15697756e-02,-1.12445159e-05,
            -1.19684450e-02,-2.98326719e-03,-1.20578352e-02,4.27098805e-03,
            1.20983357e-02],
            'state_std': [6.941581,6.087227,0.15410224,0.41530272,0.34299192,0.3524159,
            0.55824274,0.44021702,0.26541287,0.43431082,0.27763572,0.44321147,
            0.30608666,0.4479635,0.2696792,0.657875,0.67843026,0.6316779,
            0.97105443,1.0052996,1.0720154,2.348003,1.7082642,2.509587,
            1.6060903,2.5049906,1.7146177,2.3726645,1.6075712],
            },
            
        'antmaze-large-diverse-v2': {
            'state_mean': [1.9886026e+01,1.3796745e+01,4.8072639e-01,5.3255069e-01,
            2.6813401e-03,5.9484825e-02,4.9550697e-02,-5.0178785e-02,
            6.8969774e-01,9.9696316e-02,-7.1118277e-01,-6.2310215e-02,
            -7.4922961e-01,9.2040952e-03,7.1105760e-01,2.4090480e-02,
            3.3919901e-02,6.4443721e-04,7.8429899e-04,-1.4999298e-04,
            7.9512720e-05,8.5397251e-04,1.2340436e-02,4.0259658e-04,
            -1.2286444e-02,-2.8556937e-03,-1.1601355e-02,3.6952118e-03,
            1.1824124e-02],
            'state_std': [9.831796,8.1313505,0.15043381,0.4208384,0.34465218,0.3284458,
            0.55359554,0.44334394,0.26635182,0.43637007,0.27854243,0.44011194,
            0.2978178,0.44678634,0.27628005,0.7010352,0.7209651,0.6332295,
            0.97313875,0.99481547,1.0467091,2.338997,1.7254114,2.5508878,
            1.6200415,2.50616,1.7290336,2.3723924,1.6211374],
            },
            
        'antmaze-large-play-v2': {
            'state_mean': [2.0035341e+01,1.3625500e+01,4.7703636e-01,5.2078587e-01,
            1.2857303e-02,3.5217941e-02,5.8709994e-02,-5.4549828e-02,
            6.8916208e-01,1.0917618e-01,-7.1401048e-01,-7.0475943e-02,
            -7.4714363e-01,7.4979076e-03,7.1162766e-01,1.3494901e-02,
            3.4026470e-02,6.2700518e-04,6.8748189e-04,-6.8727101e-04,
            -5.9838779e-04,1.1318382e-03,1.2137439e-02,-4.9177784e-04,
            -1.2507514e-02,-2.5916169e-03,-1.1003339e-02,4.2417785e-03,
            1.1692353e-02],
            'state_std': [9.593811,7.983901,0.1520066,0.4318,0.34278414,0.34658837,
            0.54730684,0.44597837,0.2659633,0.43586802,0.28150505,0.44060528,
            0.2982533,0.44683707,0.27695337,0.702657,0.7144281,0.6279691,
            0.9666906,0.9847204,1.0362898,2.3164878,1.7009965,2.5340533,
            1.591278,2.4771726,1.718799,2.3607903,1.5971584],
            },

    }