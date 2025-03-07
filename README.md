# MSc-Thesis 🤖 🎓
Testing the limits of LLMs in multilevel reasoning tasks. Anything and everything related to my MSc Thesis, basic usage and Reinforcement Learning. The work presented here is carried out by me under the tutelage of Dr. Gemma Bel Enguix.

____

This work focuses on testing LLMs reasoning abilites based on reinforcement learning adjustments. A Natural Languague (NL) problem is presented, a NL -> Logic translation is done, inference is carried out by each LLM, and a Logic -> NL translation is done to finalize the workflow. This workflow is based on [Logic-LM](https://github.com/teacherpeterpan/Logic-LLM), but instead of using a Self-Refining module, adjustments are done using Reinforcement Learning.

Baselines are generated for propositional logic, first-order logic, and non-monotonic reasoning. These baselines are compared with state-of-the-art aligned versions of LLMs (alignment is carried out using standard RLHF algorithms: PPO, DPO, GRPO). The alignment is caried out only using a particular set of logic probelms (eg. 3 models are obtained: _LLM-PropLog-Alinged_, _LLM-FOL-Aligned_, _LLM-NonMon-Aligned_), and is then tested in order to verify which alignment can generalize better to various contexts. 
