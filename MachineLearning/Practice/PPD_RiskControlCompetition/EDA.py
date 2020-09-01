# UserInfo_8清洗处理，处理后非重复项计数减小到400
data1['UserInfo_8']=[s[:-1] if s.find('市')>0 else s[:] for s in data1.UserInfo_8]