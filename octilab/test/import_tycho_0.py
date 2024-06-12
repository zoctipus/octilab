from omni.isaac.kit import SimulationApp
kit = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni.kit.commands
from omni.isaac.core import World
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools

stage = omni.usd.get_context().get_stage()
# setting up import configuration:
status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
import_config.set_fix_base(False)
import_config.set_make_default_prim(False)
my_world = World(stage_units_in_meters=1.0)
# import MJCF
omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path="/home/octipus/Projects/FineOrbit/tycho_env-master/tycho_env/assets/hebi_rope2.xml",
    import_config=import_config,
)

stage = omni.usd.get_context().get_stage()

# enable physics
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))

# set gravity
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(981.0)

while True:
    kit.update()