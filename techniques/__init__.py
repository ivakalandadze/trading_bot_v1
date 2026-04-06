from .base_technique import TechniqueResult, BaseTechnique
from .t1_screener    import GoldmanScreener
from .t2_dcf         import MorganStanleyDCF
from .t3_risk        import BridgewaterRisk
from .t4_earnings    import JPMorganEarnings
from .t5_portfolio   import BlackRockPortfolio
from .t6_technical   import CitadelTechnical
from .t7_dividend    import HarvardDividend
from .t8_competitive import BainCompetitive
from .t9_patterns    import RenaissancePatterns
from .t10_macro      import McKinseyMacro

ALL_TECHNIQUES = [
    GoldmanScreener,
    MorganStanleyDCF,
    BridgewaterRisk,
    JPMorganEarnings,
    BlackRockPortfolio,
    CitadelTechnical,
    HarvardDividend,
    BainCompetitive,
    RenaissancePatterns,
    McKinseyMacro,
]
