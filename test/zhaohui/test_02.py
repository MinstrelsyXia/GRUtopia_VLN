import isaacsim
_app = omni.kit.app.get_app()
import omni.usd
import time

start = time.time()
stage = omni.usd.get_context().new_stage()
end = time.time()
print(f"duration: {end - start} s")