Changelog
---------

0.1.3 (2024-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`octilab.envs.mdp.actions.MultiConstraintsDifferentialInverseKinematicsActionCfg`


Changed
^^^^^^^
* cleaned, memory preallocated :class:`octilab.device.rokoko_udp_receiver.Rokoko_Glove` so it is much more readable and efficient


0.1.2 (2024-06-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`octilab.envs.mdp.actions.MultiConstraintsDifferentialInverseKinematicsActionCfg`


Changed
^^^^^^^
* Removed duplicate functions in :class:`octilab.envs.mdp.actions.actions_cfg` already defined in Isaac lab
* Removed :file:`octilab.envs.mdp.actions.binary_joint_actions.py` as it completely duplicates Isaac lab implementation
* Removed :file:`octilab.envs.mdp.actions.joint_actions.py` as it completely duplicates Isaac lab implementation
* Removed :file:`octilab.envs.mdp.actions.non_holonomic_actions.py` as it completely duplicates Isaac lab implementation
* Cleaned :class:`octilab.controllers.differential_ik.DifferentialIKController`

0.1.1 (2024-06-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Rokoko smart glove device reading
* separation of :class:`octilab.envs.mdp.actions.MultiConstraintDifferentialInverseKinematicsAction` 
  from :class:`omni.isaac.lab.envs.mdp.actions.DifferentialInverseKinematicsAction`

* separation of :class:`octilab.envs.mdp.actions.MultiConstraintDifferentialIKController` 
  from :class:`omni.isaac.lab.envs.mdp.actions.DifferentialIKController`

* separation of :class:`octilab.envs.mdp.actions.MultiConstraintDifferentialIKControllerCfg` 
  from :class:`omni.isaac.lab.envs.mdp.actions.DifferentialIKControllerCfg`


Changed
^^^^^^^
* Changed :func:`octilab.envs.mdp.events.reset_tycho_to_default` to :func:`octilab.envs.mdp.events.reset_robot_to_default`
* Changed :func:`octilab.envs.mdp.events.update_joint_positions` to :func:`octilab.envs.mdp.events.update_joint_target_positions_to_current`
* Removed unnecessary import in :class:`octilab.envs.mdp.events`
* Removed unnecessary import in :class:`octilab.envs.mdp.rewards`
* Removed unnecessary import in :class:`octilab.envs.mdp.terminations`


Updated
^^^^^^^

* Updated :meth:`octilab.envs.DeformableBasedEnv.__init__` up to date with :meth:`omni.isaac.lab.envs.ManagerBasedEnv.__init__`
* Updated :class:`octilab.envs.HebiRlEnvCfg` to :class:`octilab.envs.OctiManagerBasedRlCfg`  
* Updated :class:`octilab.envs.HebiRlEnv` to :class:`octilab.envs.OctiManagerBasedRl`


0.1.0 (2024-06-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Performed octilab refactorization. Tested to work alone, and also with tycho
* Updated README Instruction
* Plan to do: check out not duplicate logic, clean up this repository.
