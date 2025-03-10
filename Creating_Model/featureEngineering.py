import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 加载四个数据集
medal_counts = pd.read_csv("../summerOly_medal_counts.csv")
hosts = pd.read_csv("../summerOly_hosts.csv")
athletes = pd.read_csv("../summerOly_athletes.csv")
medal_counts['NOC'] = medal_counts['NOC'].str.strip()

country_to_noc = {
    'United States': 'USA',
    'Greece': 'GRE',
    'Germany': 'GER',
    'France': 'FRA',
    'Great Britain': 'GBR',
    'Hungary': 'HUN',
    'Austria': 'AUT',
    'Australia': 'AUS',
    'Denmark': 'DEN',
    'Switzerland': 'SUI',
    'Mixed team': 'ZZX',  # 非正式队伍，通常表示联合队
    'Belgium': 'BEL',
    'Italy': 'ITA',
    'Cuba': 'CUB',
    'Canada': 'CAN',
    'Spain': 'ESP',
    'Luxembourg': 'LUX',
    'Norway': 'NOR',
    'Netherlands': 'NED',
    'India': 'IND',
    'Bohemia': 'BOH',  # 现捷克的一部分，历史代表
    'Sweden': 'SWE',
    'Australasia': 'ANZ',  # 过去澳大利亚和新西兰的联合队
    'Russian Empire': 'RU1',  # 非正式缩写
    'Finland': 'FIN',
    'South Africa': 'RSA',
    'Estonia': 'EST',
    'Brazil': 'BRA',
    'Japan': 'JPN',
    'Czechoslovakia': 'TCH',  # 捷克斯洛伐克，历史队伍
    'New Zealand': 'NZL',
    'Yugoslavia': 'YUG',
    'Argentina': 'ARG',
    'Uruguay': 'URU',
    'Poland': 'POL',
    'Haiti': 'HAI',
    'Portugal': 'POR',
    'Romania': 'ROU',
    'Egypt': 'EGY',
    'Ireland': 'IRL',
    'Chile': 'CHI',
    'Philippines': 'PHI',
    'Mexico': 'MEX',
    'Latvia': 'LAT',
    'Turkey': 'TUR',
    'Jamaica': 'JAM',
    'Peru': 'PER',
    'Ceylon': 'CEY',  # 现斯里兰卡
    'Trinidad and Tobago': 'TTO',
    'Panama': 'PAN',
    'South Korea': 'KOR',
    'Iran': 'IRI',
    'Puerto Rico': 'PUR',
    'Soviet Union': 'URS',
    'Lebanon': 'LBN',
    'Bulgaria': 'BUL',
    'Venezuela': 'VEN',
    'United Team of Germany': 'EUA',  # 1956-1964年的东西德联合队
    'Iceland': 'ISL',
    'Pakistan': 'PAK',
    'Bahamas': 'BAH',
    'Ethiopia': 'ETH',
    'Formosa': 'TPE',  # 现为中华台北
    'Ghana': 'GHA',
    'Morocco': 'MAR',
    'Singapore': 'SGP',
    'British West Indies': 'BWI',  # 历史联合队
    'Iraq': 'IRQ',
    'Tunisia': 'TUN',
    'Kenya': 'KEN',
    'Nigeria': 'NGR',
    'East Germany': 'GDR',
    'West Germany': 'FRG',
    'Mongolia': 'MGL',
    'Uganda': 'UGA',
    'Cameroon': 'CMR',
    'Taiwan': 'TPE',  # 中华台北
    'North Korea': 'PRK',
    'Colombia': 'COL',
    'Niger': 'NIG',
    'Bermuda': 'BER',
    'Thailand': 'THA',
    'Zimbabwe': 'ZIM',
    'Tanzania': 'TAN',
    'Guyana': 'GUY',
    'China': 'CHN',
    'Ivory Coast': 'CIV',
    'Syria': 'SYR',
    'Algeria': 'ALG',
    'Chinese Taipei': 'TPE',
    'Dominican Republic': 'DOM',
    'Zambia': 'ZAM',
    'Suriname': 'SUR',
    'Costa Rica': 'CRC',
    'Indonesia': 'INA',
    'Netherlands Antilles': 'AHO',  # 历史队伍
    'Senegal': 'SEN',
    'Virgin Islands': 'ISV',
    'Djibouti': 'DJI',
    'Unified Team': 'EUN',  # 1992年的苏联继承国联合队
    'Lithuania': 'LTU',
    'Namibia': 'NAM',
    'Croatia': 'CRO',
    'Independent Olympic Participants': 'IOP',  # 独立运动员
    'Israel': 'ISR',
    'Slovenia': 'SLO',
    'Malaysia': 'MAS',
    'Qatar': 'QAT',
    'Russia': 'RUS',
    'Ukraine': 'UKR',
    'Czech Republic': 'CZE',
    'Kazakhstan': 'KAZ',
    'Belarus': 'BLR',
    'FR Yugoslavia': 'SCG',  # 塞尔维亚和黑山
    'Slovakia': 'SVK',
    'Armenia': 'ARM',
    'Burundi': 'BDI',
    'Ecuador': 'ECU',
    'Hong Kong': 'HKG',
    'Moldova': 'MDA',
    'Uzbekistan': 'UZB',
    'Azerbaijan': 'AZE',
    'Tonga': 'TGA',
    'Georgia': 'GEO',
    'Mozambique': 'MOZ',
    'Saudi Arabia': 'KSA',
    'Sri Lanka': 'SRI',
    'Vietnam': 'VIE',
    'Barbados': 'BAR',
    'Kuwait': 'KUW',
    'Kyrgyzstan': 'KGZ',
    'Macedonia': 'MKD',
    'United Arab Emirates': 'UAE',
    'Serbia and Montenegro': 'SCG',
    'Paraguay': 'PAR',
    'Eritrea': 'ERI',
    'Serbia': 'SRB',
    'Tajikistan': 'TJK',
    'Samoa': 'SAM',
    'Sudan': 'SUD',
    'Afghanistan': 'AFG',
    'Mauritius': 'MRI',
    'Togo': 'TOG',
    'Bahrain': 'BRN',
    'Grenada': 'GRN',
    'Botswana': 'BOT',
    'Cyprus': 'CYP',
    'Gabon': 'GAB',
    'Guatemala': 'GUA',
    'Montenegro': 'MNE',
    'Independent Olympic Athletes': 'IOA',
    'Fiji': 'FIJ',
    'Jordan': 'JOR',
    'Kosovo': 'KOS',
    'ROC': 'ROC',  # 俄罗斯奥委会
    'San Marino': 'SMR',
    'North Macedonia': 'MKD',
    'Turkmenistan': 'TKM',
    'Burkina Faso': 'BUR',
    'Saint Lucia': 'LCA',
    'Dominica': 'DMA',
    'Albania': 'ALB',
    'Cabo Verde': 'CPV',
    'Refugee Olympic Team': 'ROT'
}
# 将国家全名映射到奥委会缩写代码
medal_counts['NOC_Code'] = medal_counts['NOC'].map(country_to_noc)
# 提取首次参赛年份（基于运动员数据）
first_participation = athletes.groupby("NOC")["Year"].min().reset_index(name="first_participation_year")
# 提取首次获奖年份（基于奖牌数据）
first_medal = medal_counts[medal_counts["Total"] > 0].groupby("NOC_Code")["Year"].min().reset_index(name="first_medal_year")
# 修改first_participation的NOC列名为NOC_Code，以便与first_medal匹配
first_participation.rename(columns={"NOC": "NOC_Code"}, inplace=True)
# 合并并计算Duration和Event
survival_data = pd.merge(first_participation, first_medal, on="NOC_Code", how="outer")
survival_data = survival_data.dropna(subset=["first_participation_year"])
# 计算Duration（转换为“届数”）
def calculate_duration(row):
    if pd.isna(row["first_medal_year"]):
        # 从未获奖：计算到2024年的总届数（假设奥运会每4年一届）
        return (2024 - row["first_participation_year"]) // 4 + 1
    else:
        # 已获奖：计算从首次参赛到首次获奖的届数
        return (row["first_medal_year"] - row["first_participation_year"]) // 4

survival_data["duration"] = survival_data.apply(calculate_duration, axis=1)
# 标记Event
survival_data["event"] = survival_data["first_medal_year"].notna().astype(int)
features = pd.DataFrame()
features = survival_data.copy()


# 国家名称映射（处理不一致问题）
country_name_map = {
    "United Kingdom": "Great Britain",
    "Japan (postponed to 2021 due to the coronavirus pandemic)":"Japan"
}
# 清理字符串中的非标准空格和多余空格
hosts["Host"] = hosts["Host"].str.replace(r'\xa0', ' ', regex=True).str.strip()
# 提取主办国家并应用名称映射
hosts["host_country"] = hosts["Host"].apply(lambda x: x.split(",")[-1].strip())
hosts["host_country"] = hosts["host_country"].replace(country_name_map)
# 合并主办国信息到 medal_counts
medal_counts = pd.merge(medal_counts, hosts[["Year", "host_country"]], on="Year", how="left")
# 判断是否为主办国
medal_counts["is_host"] = (medal_counts["NOC"] == medal_counts["host_country"]).astype(int)


# 参与度特征
participation_stats = athletes.groupby('NOC').agg({
    'Year': 'nunique',  # 参加届数
    'Name': 'count',  # 历史总运动员数
    'Event': 'nunique',  # 历史参与项目数量
    'Sport': 'nunique'  # 历史参与大项数量
}).reset_index()
participation_stats.columns = ['NOC_Code', 'total_appearances', 'total_athletes', 'unique_events', 'unique_sports']
# 计算每届平均值
participation_stats['athletes_per_games'] = participation_stats['total_athletes'] / participation_stats[
    'total_appearances']
participation_stats['events_per_games'] = participation_stats['unique_events'] / participation_stats[
    'total_appearances']
features = features.merge(participation_stats, on='NOC_Code', how='left')

athlete_structure = athletes.groupby('NOC').agg({
        'Sex': lambda x: (x == 'F').mean()  # 女性运动员比例
    }).reset_index()
athlete_structure.columns = ['NOC_Code', 'female_ratio']
features = features.merge(athlete_structure, on='NOC_Code', how='left')


# 定义修正后的 calculate_growth 函数
def calculate_growth(group):
    # 获取该国家参赛的所有年份
    years = group['Year'].unique()
    if len(years) <= 1:
        # 如果参赛年份少于等于1，无法计算增长率
        return 0
    # 排序年份
    years = sorted(years)
    # 获取该国家在首次参赛年份的运动员数量
    first_count = group[group['Year'] == years[0]]['Name'].count()
    # 获取该国家在最后一次参赛年份的运动员数量
    last_count = group[group['Year'] == years[-1]]['Name'].count()
    # 计算增长率，如果首次参赛人数为0，返回0
    return (last_count - first_count) / first_count if first_count > 0 else 0
# 计算每个国家的运动员增长率
growth_stats = athletes.groupby('NOC').apply(calculate_growth).reset_index(name='athlete_growth')
growth_stats.rename(columns={'NOC': 'NOC_Code'}, inplace=True)
# 将增长率特征合并到 features 数据集中
features = features.merge(growth_stats, on='NOC_Code', how='left')


host_stats = medal_counts.groupby('NOC_Code').agg({
    'is_host': 'sum'  # 主办次数
}).reset_index()
# 给列命名为 'host_count'
host_stats.rename(columns={'is_host': 'host_count'}, inplace=True)
features = features.merge(host_stats, on='NOC_Code', how='left')
features["host_count"] = features["host_count"].fillna(0)


# 计算每个国家运动员参与的项目多样性
athlete_diversity = athletes.groupby(['NOC', 'Name'])['Event'].nunique().reset_index()
diversity_stats = athlete_diversity.groupby('NOC').agg({
    'Event': 'mean'  # 平均每个运动员参与的项目数
}).reset_index()
diversity_stats.columns = ['NOC_Code', 'events_per_athlete']
features = features.merge(diversity_stats, on='NOC_Code', how='left')
