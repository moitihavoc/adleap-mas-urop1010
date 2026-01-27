from setuptools import setup

#####
# Major Environment
#####
setup(name='AdhocReasoningEnv',
      version='1.0.0',
      install_requires=['gym']
)

#####
# Toy Problems Environments
#####
setup(name='TigerEnv',
      version='2.0.0',
      install_requires=['gym','numpy'],
)

setup(name='MazeEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

setup(name='RockSampleEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

setup(name='TagEnv',
      version='1.0.0',
      install_requires=['gym','numpy']
)

setup(name='LaserTagEnv',
      version='1.0.0',
      install_requires=['gym','numpy']
)

#####
# Ad-hoc Teamwork Environment
#####
setup(name='LevelForagingEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

setup(name='CaptureEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

#####
# Realistic Scenarios
#####
setup(name='SmartFireBrigadeEnv',
      version='1.0.0',
      install_requires=['gym','numpy']
)
setup(name='TradeStockEnv',
      version='1.0.0',
      install_requires=['gym','numpy','pandas','sklearn','statsmodels','scipy']
)
setup(name='InfiltrationEnv',
      version='1.0.0',
      install_requires=['gym','numpy','pandas']
)

#####
# Card Games
#####
setup(name='TrucoEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

#####
# Integrated Games (third source)
#####
setup(name='ChessEnv',
      version='1.0.0',
      install_requires=['gym','chess','numpy','PyQt5']
)