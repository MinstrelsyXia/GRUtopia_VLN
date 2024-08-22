from vln.src.local_nav.sementic_map import Semantic_Mapping

# arguments
args=get_args()
sem_map_module = Semantic_Mapping(args).to(device)
sem_map_module.eval()
