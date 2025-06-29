import quaternion

class PoseConfig:
    poses = {
        "path0": {
            "position": [3.3094482, 0.13348278, 7.6297503],
            "quaternion": quaternion.quaternion(-0.00872616190463305, 0, -0.999961912631989, 0)
        },
        "path1": {
            "position": [-1.459727, 0.11279388, 9.546749],
            "quaternion": quaternion.quaternion(-0.0174519009888172, 0, -0.999847710132599, 0)
        },
        "path2": {
            "position": [-4.674316, 0.20991862, 16.165129],
            "quaternion": quaternion.quaternion(0.0261774249374866, 0, -0.999657332897186, 0)
        },
        "path3": {
            "position": [3.482629, 0.07714556, 17.708155],
            "quaternion": quaternion.quaternion(0.999048233032227, 0, 0.0436198562383652, 0)
        },
        "path4": {
            "position": [4.943766, 0.18818882, 8.543601],
            "quaternion": quaternion.quaternion(0.0523367486894131, 0, -0.998629450798035, 0)
        },
        "path4l": {
            "position": [3.0082424, 0.20991862, 14.09007],
            "quaternion": quaternion.quaternion(-0.113202415406704, 0, -0.993571996688843, 0)
        },
        "path13": {
            "position": [2.5130017, 0.20991862, 15.108349],
            "quaternion": quaternion.quaternion(-1, 0, -1.03842467069626e-06, 0)
        },
        "path13l": {
            "position": [2.6239955, 0.10881426, 7.6125574],
            "quaternion": quaternion.quaternion(-0.902584910392761, 0, -0.430512011051178, 0)
        },
        "path19": {
            "position": [-6.4327126, 0.01625727, 7.429369],
            "quaternion": quaternion.quaternion(0.00872724782675505, 0, -0.999961912631989, 0)
        },
        "path19l": {
            "position": [-4.1785393, 0.33463737, 17.958515],
            "quaternion": quaternion.quaternion(0.0523366071283817, 0, -0.998629510402679, 0)
        },
        "path21": {
            "position": [-6.4327126, 0.01625727, 7.429369],
            "quaternion": quaternion.quaternion(0.00872724782675505, 0, -0.999961912631989, 0)
        },
        "path26": {
            "position": [-3.97578, 0.32081732, 17.921337],
            "quaternion": quaternion.quaternion(0.0436200611293316, 0, -0.999048233032227, 0)
        },
        "path27": {
            "position": [-3.97578, 0.32081732, 17.921337],
            "quaternion": quaternion.quaternion(0.0436200611293316, 0, -0.999048233032227, 0)
        },
        "path28": {
            "position": [-3.97578, 0.32081732, 17.921337],
            "quaternion": quaternion.quaternion(0.0436200611293316, 0, -0.999048233032227, 0)
        },
        "path29": {
            "position": [-6.4327126, 0.01625727, 7.429369],
            "quaternion": quaternion.quaternion(0.00872724782675505, 0, -0.999961912631989, 0)
        },
        "path30": {
            "position": [-0.7335855, 0.10477532, 9.280411],
            "quaternion": quaternion.quaternion(0.707107365131378, 0, -0.707106232643127, 0)
        },
        "path31": {
            "position": [-3.6873128, 0.2674058, 18.825111],
            "quaternion": quaternion.quaternion(0.00872755330055952, 0, -0.999961912631989, 0)
        },
        "path32": {
            "position": [2.0086925, 0.20991862, 14.936816],
            "quaternion": quaternion.quaternion(0.622515380382538, 0, -0.782607614994049, 0)
        },
        "path33": {
            "position": [-5.609331, 0.20991862, 14.444713],
            "quaternion": quaternion.quaternion(0.713250994682312, 0, -0.700908720493317, 0)
        },
        "path-34": {
            "position": [-2.8017743, 0.1507751, 20.25784],
            "quaternion": quaternion.quaternion(0.649448871612549, 0, -0.760405361652374, 0),
            "notes": "near n:69 tm:56 as experiment to reach n:31 tm:56"
        },
        "path-35": {
            "position": [0.24059151, 0.13737503, 20.931145],
            "quaternion": quaternion.quaternion(0.649448871612549, 0, -0.760405302047729, 0),
            "notes": "near n:71 tm:56 as experiment to reach n:31 tm:56"
        },
        "path-36": {
            "position": [-2.5960038, 0.20991862, 14.566516],
            "quaternion": quaternion.quaternion(0.580703854560852, 0, -0.814114809036255, 0),
            "notes": "near n:71 tm:56 as experiment to reach n:31 tm:56"
        },
        "VG-0": {
            "position": [2.9874349, 0.17669876, -1.4243687],
            "quaternion": quaternion.quaternion(0.0261764898896217, 0, 0.999657332897186, 0)
        },
        "VG-1": {
            "position": [3.219628, 0.17669876, 1.5275097],
            "quaternion": quaternion.quaternion(0.933580160140991, 0, 0.358368694782257, 0)
        },
        "VG-2": {
            "position": [2.0534012, 0.17669876, 0.5477569],
            "quaternion": quaternion.quaternion(-0.809015989303589, 0, -0.587786674499512, 0),
            "notes": "short path from the left to the center"
        },
        "VG-3": {
            "position": [3.137889, 0.17669876, -1.22471],
            "quaternion": quaternion.quaternion(0.0261764861643314, 0, 0.999657332897186, 0),
            "notes": "short path from the left to the center"
        },
        "origin": {
            "position": [0.0, 0.0, 0.0],
            "quaternion": quaternion.quaternion(0.933580160140991, 0, 0.358368694782257, 0)
        },
        "app-0": {
            "position": [5.9790516, -1.60025, -2.29948],
            "quaternion": quaternion.quaternion(0.681998014450073, 0, 0.731354057788849, 0)
        },
        "app-1": {
            "position": [0.15647297, -1.60025, 6.174869],
            "quaternion": quaternion.quaternion(-0.978147625923157, 0, 0.20791158080101, 0)
        },
        "app-2": {
            "position": [4.6671658, -1.60025, 1.1856903],
            "quaternion": quaternion.quaternion(0.743144452571869, 0, 0.669131100177765, 0)
        },
        "rep-0": {
            "position": [-4.9248104, 0.072447, 0.19702175],
            "quaternion": quaternion.quaternion(-0.700908660888672, 0, -0.713251054286957, 0)
        },
        "rep-1": {
            "position": [-9.926618, 0.072447, 0.8911567],
            "quaternion": quaternion.quaternion(0.958820044994354, 0, -0.284014403820038, 0)
        },
        "rep-2": {
            "position": [-10.321852, 0.072447, -2.5800464],
            "quaternion": quaternion.quaternion(0.130527123808861, 0, -0.991444826126099, 0)
        },
        "rep-3": {
            "position": [-5.2826066, 0.072447, -2.6461887],
            "quaternion": quaternion.quaternion(-0.414693921804428, 0, 0.909961044788361, 0)
        },
        "rep-4": {
            "position": [0.68483937, 0.072447, -3.7973828],
            "quaternion": quaternion.quaternion(-0.078458659350872, 0, -0.996917366981506, 0)
        },
        "rep-5": {
            "position": [4.110597, 0.072447, 1.2330898],
            "quaternion": quaternion.quaternion(0.782607853412628, 0, 0.622515141963959, 0)
        },
        "rep-6": {
            "position": [2.2813768, 0.072447, -4.2409663],
            "quaternion": quaternion.quaternion(-0.0436184406280518, 0, -0.999048352241516, 0)
        },
        "rep-7": {
            "position": [-8.621148 ,  0.072447 , -0.1713194],
            "quaternion": quaternion.quaternion(-0.999961912631989, 0, -0.00872752815485001, 0)
        },
        "castle-0": {
            "position": [-3.7636783 ,  0.28988877, 18.349354],
            "quaternion": quaternion.quaternion(0.0610489770770073, 0, -0.998134791851044, 0)
        },
        "castle-1": {
            "position": [3.0325296 , 0.13065091, 9.493785],
            "quaternion": quaternion.quaternion(0.0087270075455308, 0, -0.999961912631989, 0)
        },
        "castle-2": {
            "position": [1.1738753 ,  0.18384442, 19.429428],
            "quaternion": quaternion.quaternion(0.902585387229919, 0, -0.430510938167572, 0)
        },
        "castle-3": {
            "position": [0.11749813, 0.08730142, 8.688587  ],
            "quaternion": quaternion.quaternion(-0.713250815868378, 0, 0.700908899307251, 0)
        },
        "rep_wc-0": {
            "position": [-5.4343696 ,  0.072447  ,  0.48555583],
            "quaternion": quaternion.quaternion(-0.737277030944824, 0, -0.675590634346008, 0)
        },
        "rep_bed_tv-0": {
            "position": [-9.832949 ,  0.072447 , -0.8266918],
            "quaternion": quaternion.quaternion(-0.887011170387268, 0, 0.461748003959656, 0)
        },
        "rep_bed_tv-1": {
            "position": [-10.200939,   0.072447,  -2.832789],
            "quaternion": quaternion.quaternion(-0.0958465114235878, 0, 0.995396137237549, 0)
        },
        "rep_kitchen-0": {
            "position": [-2.6727195,  0.072447 , -1.4695278],
            "quaternion": quaternion.quaternion(-0.636078834533691, 0, 0.771624147891998, 0)
        },
        "rep_dinning-0": {
            "position": [ 0.67513055,  0.072447  , -3.2634013 ],
            "quaternion": quaternion.quaternion(0.0784583315253258, 0, 0.996917426586151, 0)
        }
    }