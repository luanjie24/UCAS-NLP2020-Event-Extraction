NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

# for BERT
CLS = '[CLS]'
SEP = '[SEP]'

# 34 event triggers
TRIGGERS = ['Business:Merge-Org',
            'Business:Start-Org',
            'Business:Declare-Bankruptcy',
            'Business:End-Org',
            'Justice:Pardon',
            'Justice:Extradite',
            'Justice:Execute',
            'Justice:Fine',
            'Justice:Trial-Hearing',
            'Justice:Sentence',
            'Justice:Appeal',
            'Justice:Convict',
            'Justice:Sue',
            'Justice:Release-Parole',
            'Justice:Arrest-Jail',
            'Justice:Charge-Indict',
            'Justice:Acquit',
            'Conflict:Demonstrate',
            'Conflict:Attack',
            'Contact:Phone-Write',
            'Contact:Meet',
            'Personnel:Start-Position',
            'Personnel:Elect',
            'Personnel:End-Position',
            'Personnel:Nominate',
            'Transaction:Transfer-Ownership',
            'Transaction:Transfer-Money',
            'Life:Marry',
            'Life:Divorce',
            'Life:Be-Born',
            'Life:Die',
            'Life:Injure',
            'Movement:Transport']

"""
    28 argument roles
    
    There are 35 roles in ACE2005 dataset, but the time-related 8 roles were replaced by 'Time' as the previous work (Yang et al., 2016).
    ['Time-At-End','Time-Before','Time-At-Beginning','Time-Ending', 'Time-Holds', 'Time-After','Time-Starting', 'Time-Within'] --> 'Time'.
"""
ARGUMENTS = ['Place',
             'Crime',
             'Prosecutor',
             'Sentence',
             'Org',
             'Seller',
             'Entity',
             'Agent',
             'Recipient',
             'Target',
             'Defendant',
             'Plaintiff',
             'Origin',
             'Artifact',
             'Giver',
             'Position',
             'Instrument',
             'Money',
             'Destination',
             'Buyer',
             'Beneficiary',
             'Attacker',
             'Adjudicator',
             'Person',
             'Victim',
             'Price',
             'Vehicle',
             'Time']

# 54 entities
ENTITIES = ['VEH:Water',
            'GPE:Nation',
            'ORG:Commercial',
            'GPE:State-or-Province',
            'Contact-Info:E-Mail',
            'Crime',
            'ORG:Non-Governmental',
            'Contact-Info:URL',
            'Sentence',
            'ORG:Religious',
            'VEH:Underspecified',
            'WEA:Projectile',
            'FAC:Building-Grounds',
            'PER:Group',
            'WEA:Exploding',
            'WEA:Biological',
            'Contact-Info:Phone-Number',
            'WEA:Chemical',
            'LOC:Land-Region-Natural',
            'WEA:Nuclear',
            'LOC:Region-General',
            'PER:Individual',
            'WEA:Sharp',
            'ORG:Sports',
            'ORG:Government',
            'ORG:Media',
            'LOC:Address',
            'WEA:Shooting',
            'LOC:Water-Body',
            'LOC:Boundary',
            'GPE:Population-Center',
            'GPE:Special',
            'LOC:Celestial',
            'FAC:Subarea-Facility',
            'PER:Indeterminate',
            'VEH:Subarea-Vehicle',
            'WEA:Blunt',
            'VEH:Land',
            'TIM:time',
            'Numeric:Money',
            'FAC:Airport',
            'GPE:GPE-Cluster',
            'ORG:Educational',
            'Job-Title',
            'GPE:County-or-District',
            'ORG:Entertainment',
            'Numeric:Percent',
            'LOC:Region-International',
            'WEA:Underspecified',
            'VEH:Air',
            'FAC:Path',
            'ORG:Medical-Science',
            'FAC:Plant',
            'GPE:Continent']

# 45 pos tags
POSTAGS = ['VBZ', 'NNS', 'JJR', 'VB', 'RBR',
           'WP', 'NNP', 'RP', 'RBS', 'VBP',
           'IN', 'UH', 'JJS', 'NNPS', 'PRP$',
           'MD', 'DT', 'WP$', 'POS', 'LS',
           'CC', 'VBN', 'EX', 'NN', 'VBG',
           'SYM', 'FW', 'TO', 'JJ', 'VBD',
           'WRB', 'CD', 'PDT', 'WDT', 'PRP',
           'RB', ',', '``', "''", ':',
           '.', '$', '#', '-LRB-', '-RRB-']
##
# event_set = set(TRIGGERS)
event_list = ['Conflict:Attack', 'Movement:Transport', 'Life:Die', 'Contact:Meet', 'Personnel:End-Position', 'Personnel:Elect', 'Life:Injure', 'Transaction:Transfer-Money', 'Contact:Phone-Write', 'Justice:Trial-Hearing', 'Justice:Charge-Indict', 'Transaction:Transfer-Ownership', 'Personnel:Start-Position', 'Justice:Sentence', 'Justice:Arrest-Jail', 'Justice:Sue', 'Life:Marry', 'Conflict:Demonstrate', 'Justice:Convict', 'Life:Be-Born', 'Justice:Release-Parole', 'Business:Declare-Bankruptcy', 'Business:End-Org', 'Justice:Appeal', 'Business:Start-Org', 'Justice:Fine', 'Life:Divorce', 'Justice:Execute', 'Business:Merge-Org', 'Personnel:Nominate', 'Justice:Acquit', 'Justice:Extradite', 'Justice:Pardon']
event_cls = []
for event in event_list:
    event_cls.append({event})