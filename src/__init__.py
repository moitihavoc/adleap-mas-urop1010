from gymnasium import register

#####
# Major Environment
#####
register(
    id='AdhocReasoningEnv-v1',
    entry_point='src.envs:AdhocReasoningEnv',
)

#####
# Toy Problems Environments
#####
register(
    id='TigerEnv-v2',
    entry_point='src.envs:TigerEnv',
    )
register(
    id='MazeEnv-v2',
    entry_point='src.envs:MazeEnv',
)

register(
    id='RockSampleEnv-v2',
    entry_point='src.envs:RockSampleEnv',
)

register(
    id='TagEnv-v1',
    entry_point='src.envs:TagEnv',
)

register(
    id='LaserTagEnv-v1',
    entry_point='src.envs:TagEnv',
)

#####
# Ad-hoc Teamwork Environment
#####
register(
    id='LevelForagingEnv-v2',
    entry_point='src.envs:LevelForagingEnv',
)
register(
    id='CaptureEnv-v2',
    entry_point='src.envs:CaptureEnv',
)


#####
# Realistic Scenarios
#####
register(
    id='SmartFireBrigadeEnv-v1',
    entry_point='src.envs:SmartFireBrigadeEnv',
)
register(
    id='TradeStockEnv-v1',
    entry_point='src.envs:TradeStockEnv',
)
register(
    id='InfiltrationEnv-v1',
    entry_point='src.envs:InfiltrationEnv',
)

#####
# Card Games
#####
register(
    id='TrucoEnv-v2',
    entry_point='src.envs:TrucoEnv',
)

#####
# Integrated Games (third source)
#####
register(
    id='ChessEnv-v1',
    entry_point='src.envs:ChessEnv',
)