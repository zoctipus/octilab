Changelog
---------


0.1.7 (2024-07-08)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Documented :class:`octi.lab.devices.rokoko_glove.RokokoGlove`,
  :class:`octi.lab.devices.rokoko_glove.RokokoGloveKeyboard`, :class:`octi.lab.devices.rokoko_glove.Se3Keyboard`



0.1.7 (2024-07-08)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Added proximal distance scaling in :class:`octi.lab.devices.rokoko_glove.RokokoGlove`
* Fixed the order checking for the :class:`octi.lab.controllers.differential_ik.MultiConstraintDifferentialIKController`


Added
^^^^^
* Added combined control that separates pose and finger joints in
  :class:`octi.lab.devices.rokoko_glove_keyboard.RokokoGloveKeyboard`


0.1.6 (2024-07-06)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* :class:`octi.lab.actuators.actuator_cfg.HebiStrategy3ActuatorCfg` added the field that scales position_p and effort_p
* :class:`octi.lab.actuators.actuator_cfg.HebiStrategy4ActuatorCfg` added the field that scales position_p and effort_p
* :class:`octi.lab.actuators.actuator_pd.py.HebiStrategy3Actuator` reflected the field that scales position_p and effort_p
* :class:`octi.lab.actuators.actuator_pd.py.HebiStrategy4Actuator` reflected the field that scales position_p and effort_p
* Improved Reuseability :class:`octi.lab.devices.rokoko_udp_receiver.Rokoko_Glove` such that the returned joint position respects the
order user inputs. Added debug visualization. Plan to add scale by knuckle width to match the leap hand knuckle width

0.1.5 (2024-07-04)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^
* :meth:`octi.lab.envs.octi_manager_based_rl.step` the actual environment update rate now becomes 
decimation square, as square allows a nice property that tuning decimation creates minimal effect on the learning 
behavior. 


0.1.4 (2024-06-29)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^
* allow user input specific tracking name :meth:`octi.lab.device.rokoko_udp_receiver.Rokoko_Glove.__init__` to address
  inefficienty when left or right has tracking is unnecessary, and future need in increasing, decreasing number of track
  parts with ease. In addition, the order which parts are outputed is now ordered by user's list input, removing the need
  of manually reorder the output when the output is fixed

0.1.3 (2024-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`octi.lab.envs.mdp.actions.MultiConstraintsDifferentialInverseKinematicsActionCfg`


Changed
^^^^^^^
* cleaned, memory preallocated :class:`octi.lab.device.rokoko_udp_receiver.Rokoko_Glove` so it is much more readable and efficient


0.1.2 (2024-06-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`octi.lab.envs.mdp.actions.MultiConstraintsDifferentialInverseKinematicsActionCfg`


Changed
^^^^^^^
* Removed duplicate functions in :class:`octi.lab.envs.mdp.actions.actions_cfg` already defined in Isaac lab
* Removed :file:`octi.lab.envs.mdp.actions.binary_joint_actions.py` as it completely duplicates Isaac lab implementation
* Removed :file:`octi.lab.envs.mdp.actions.joint_actions.py` as it completely duplicates Isaac lab implementation
* Removed :file:`octi.lab.envs.mdp.actions.non_holonomic_actions.py` as it completely duplicates Isaac lab implementation
* Cleaned :class:`octi.lab.controllers.differential_ik.DifferentialIKController`

0.1.1 (2024-06-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Rokoko smart glove device reading
* separation of :class:`octi.lab.envs.mdp.actions.MultiConstraintDifferentialInverseKinematicsAction` 
  from :class:`omni.isaac.lab.envs.mdp.actions.DifferentialInverseKinematicsAction`

* separation of :class:`octi.lab.envs.mdp.actions.MultiConstraintDifferentialIKController` 
  from :class:`omni.isaac.lab.envs.mdp.actions.DifferentialIKController`

* separation of :class:`octi.lab.envs.mdp.actions.MultiConstraintDifferentialIKControllerCfg` 
  from :class:`omni.isaac.lab.envs.mdp.actions.DifferentialIKControllerCfg`


Changed
^^^^^^^
* Changed :func:`octi.lab.envs.mdp.events.reset_tycho_to_default` to :func:`octi.lab.envs.mdp.events.reset_robot_to_default`
* Changed :func:`octi.lab.envs.mdp.events.update_joint_positions` to :func:`octi.lab.envs.mdp.events.update_joint_target_positions_to_current`
* Removed unnecessary import in :class:`octi.lab.envs.mdp.events`
* Removed unnecessary import in :class:`octi.lab.envs.mdp.rewards`
* Removed unnecessary import in :class:`octi.lab.envs.mdp.terminations`


Updated
^^^^^^^

* Updated :meth:`octi.lab.envs.DeformableBasedEnv.__init__` up to date with :meth:`omni.isaac.lab.envs.ManagerBasedEnv.__init__`
* Updated :class:`octi.lab.envs.HebiRlEnvCfg` to :class:`octi.lab.envs.OctiManagerBasedRlCfg`  
* Updated :class:`octi.lab.envs.HebiRlEnv` to :class:`octi.lab.envs.OctiManagerBasedRl`


0.1.0 (2024-06-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Performed octi.lab refactorization. Tested to work alone, and also with tycho
* Updated README Instruction
* Plan to do: check out not duplicate logic, clean up this repository.