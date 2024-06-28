Changelog
---------

0.1.1 (2024-06-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Rokoko smart glove device reading


Changed
^^^^^^^
* Changed :func:`octilab.envs.mdp.events.reset_tycho_to_default` to :func:`octilab.envs.mdp.events.reset_robot_to_default`
* Changed :func:`octilab.envs.mdp.events.update_joint_positions` to :func:`octilab.envs.mdp.events.update_joint_target_positions_to_current`
* Removed unnecessary import in :class:`octilab.envs.mdp.events`
* Removed unnecessary import in :class:`octilab.envs.mdp.rewards`
* Removed unnecessary import in :class:`octilab.envs.mdp.terminations`


Updated
^^^^^^^

* Updated :func:`octilab.envs.DeformableBasedEnv.__init__` up to date with :class:`omni.isaac.lab.envs.ManagerBasedEnv.__init__`
* Updated :class:`octilab.envs.HebiRlEnvCfg` to :class:`octilab.envs.OctiManagerBasedRlCfg`  
* Updated :class:`octilab.envs.HebiRlEnv` to :class:`octilab.envs.OctiManagerBasedRl`


0.1.0 (2024-06-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Performed octilab refactorization. Tested to work alone, and also with tycho
* Updated README Instruction
* Plan to do: check out not duplicate logic, clean up this repository.
