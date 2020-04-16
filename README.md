# Learning_Reward_Function
Attempt to learn reward function for autonomous earth moving vehicle using demonstrations

**Run on Mountain Car**
1. To run once (about 15 minutes)
`python3.6 examples/upr_tomountaincar.py`
2. To run _n_ times 
In examples/upr_tomountaincar.py 
i. Comment `run_once()` line 77
ii. Uncomment `run_multiple(100)` line 78 where 100 could be changed to n

**Run on Avant Data**
1. Save demonstration CSVs to examples/data
2. Edit line 73 in examples/upr_toavant and add folders containing 
   demonstration CSVs and set number of segments/clusters
3. Run `python3.6 examples/upr_toavant.py`
_Note: To Test one file - Uncomment and edit lines 78-80_
