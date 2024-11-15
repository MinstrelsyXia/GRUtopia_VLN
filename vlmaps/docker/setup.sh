export https_proxy=https://xiaxinyuan:OE6gf5X1v0JkSjKDOoUsVZhCdBbf0mdwfWO2kvWSlKj9L0Jwcfb9ff7snMkk@blsc-proxy.pjlab.org.cn:13128
export http_proxy=https://xiaxinyuan:OE6gf5X1v0JkSjKDOoUsVZhCdBbf0mdwfWO2kvWSlKj9L0Jwcfb9ff7snMkk@blsc-proxy.pjlab.org.cn:13128

pip install omegaconf
pip install -r requirements/requirements.txt
pip install -r requirements/failed_packages.txt
pip install httpx==0.25.2 httpcore==1.0.5 # else openai will fail
