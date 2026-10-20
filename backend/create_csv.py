import os
csv_content = """state,district,latitude,longitude
Andhra Pradesh,Guntur,16.31,80.44
Andhra Pradesh,Krishna,16.61,81.08
Andhra Pradesh,Nellore,14.44,79.99
Andhra Pradesh,Kadapa,14.47,78.82
Andhra Pradesh,Kurnool,15.83,78.04
Andhra Pradesh,Chittoor,13.21,79.10
Andhra Pradesh,East Godavari,17.08,82.24
Andhra Pradesh,West Godavari,16.92,81.33
Andhra Pradesh,Visakhapatnam,17.69,83.22
Andhra Pradesh,Vizianagaram,18.12,83.39
Andhra Pradesh,Srikakulam,18.27,83.90
Andhra Pradesh,Prakasam,15.94,79.55
Andhra Pradesh,Anantapur,14.68,77.60
Assam,Kamrup,26.14,91.76
Assam,Nagaon,26.35,92.68
Assam,Darrang,26.47,92.06
Assam,Sonitpur,26.61,92.80
Assam,Lakhimpur,27.24,94.10
Assam,Golagi,26.80,93.97
Assam,Tinsukia,27.49,95.36
Bihar,Patna,25.59,85.14
Bihar,Gaya,24.79,85.01
Bihar,Muzaffarpur,26.12,85.39
Bihar,Darbhanga,26.17,85.90
Bihar,Bhagalpur,25.24,86.98
Bihar,Purnia,25.78,87.47
Bihar,Samastipur,25.87,85.78
Bihar,Saran,25.91,84.74
Bihar,Begusarai,25.42,86.13
Bihar,Siwan,26.22,84.36
Bihar,Rohtas,24.97,83.99
Bihar,Nalanda,25.11,85.45
Gujarat,Ahmedabad,23.02,72.57
Gujarat,Surat,21.17,72.83
Gujarat,Rajkot,22.30,70.80
Gujarat,Vadodara,22.31,73.18
Gujarat,Bhavnagar,21.77,72.15
Gujarat,Jamnagar,22.47,70.07
Gujarat,Junagadh,21.52,70.46
Gujarat,Anand,22.56,72.95
Gujarat,Mehsana,23.60,72.39
Gujarat,Banaskantha,24.17,72.43
Gujarat,Sabarkantha,23.37,73.00
Gujarat,Amreli,21.60,71.22
Gujarat,Kheda,22.75,72.69
Gujarat,Gandhinagar,23.21,72.68
Gujarat,Kachchh,23.73,70.00
Gujarat,Porbandar,21.64,69.63
Haryana,Ambala,30.38,76.78
Haryana,Kurukshetra,29.97,76.84
Haryana,Karnal,29.68,76.99
Haryana,Rohtak,28.90,76.58
Haryana,Hisar,29.15,75.72
Haryana,Sirsa,29.54,75.03
Haryana,Faridabad,28.41,77.31
Haryana,Gurgaon,28.46,77.03
Haryana,Bhiwani,28.79,76.13
Haryana,Sonipat,28.99,77.02
Haryana,Jind,29.32,76.32
Haryana,Fatehabad,29.52,75.45
Karnataka,Bangalore,12.97,77.59
Karnataka,Mysore,12.31,76.65
Karnataka,Dharwad,15.46,75.01
Karnataka,Belgaum,15.85,74.50
Karnataka,Gulbarga,17.33,76.82
Karnataka,Tumkur,13.34,77.10
Karnataka,Shimoga,13.93,75.57
Karnataka,Davangere,14.47,75.92
Karnataka,Hassan,13.00,76.10
Karnataka,Mandya,12.52,76.90
Karnataka,Raichur,16.20,77.36
Karnataka,Bidar,17.91,77.52
Karnataka,Bijapura,16.83,75.72
Karnataka,Chikmagalur,13.32,75.78
Karnataka,Chitradurga,14.23,76.40
Karnataka,Bellary,15.14,76.92
Karnataka,Kolar,13.14,78.13
Karnataka,Ramanagara,12.72,77.28
Karnataka,Kodagu,12.42,75.74
Madhya Pradesh,Indore,22.72,75.86
Madhya Pradesh,Bhopal,23.26,77.41
Madhya Pradesh,Jabalpur,23.18,79.99
Madhya Pradesh,Gwalior,26.22,78.18
Madhya Pradesh,Ujjain,23.18,75.77
Madhya Pradesh,Sagar,23.84,78.74
Madhya Pradesh,Rewa,24.53,81.30
Madhya Pradesh,Satna,24.57,80.83
Madhya Pradesh,Ratiam,23.33,75.04
Madhya Pradesh,Hoshangabad,22.75,77.73
Madhya Pradesh,Vidisha,23.52,77.81
Madhya Pradesh,Chhindwara,22.06,78.93
Madhya Pradesh,Morena,26.50,78.00
Madhya Pradesh,Damoh,23.83,79.44
Madhya Pradesh,Shivpuri,25.42,77.66
Madhya Pradesh,Mandsaur,24.07,75.07
Madhya Pradesh,Devas,22.96,76.05
Madhya Pradesh,Shajapur,23.43,76.28
Madhya Pradesh,Guuna,24.65,77.31
Madhya Pradesh,Khargone,21.83,75.62
Madhya Pradesh,Khandwa,21.82,76.35
Madhya Pradesh,Betul,21.91,77.90
Madhya Pradesh,Sheopur,25.67,76.70
Maharashtra,Nashik,20.01,73.79
Maharashtra,Nagpur,21.15,79.09
Maharashtra,Pune,18.52,73.86
Maharashtra,Aurangabad,19.88,75.34
Maharashtra,Amravati,20.93,77.75
Maharashtra,Solapur,17.69,75.90
Maharashtra,Latur,18.40,76.56
Maharashtra,Nanded,19.16,77.30
Maharashtra,Kolhapur,16.71,74.24
Maharashtra,Satara,17.69,74.00
Maharashtra,Sangli,16.86,74.57
Maharashtra,Ahmednagar,19.09,74.74
Maharashtra,Jalgaon,21.01,75.56
Maharashtra,Dhule,20.90,74.78
Maharashtra,Osmanaabad,18.18,76.04
Maharashtra,Bid,18.99,75.76
Maharashtra,Buldana,20.53,76.18
Maharashtra,Yavatmal,20.39,78.12
Maharashtra,Akola,20.71,77.00
Maharashtra,Wardha,20.74,78.60
Maharashtra,Gadchiroli,19.80,80.07
Maharashtra,Chandrapur,19.95,79.30
Maharashtra,Raigad,18.52,73.18
Maharashtra,Thane,19.22,73.00
Maharashtra,Ratnagiri,16.99,73.30
Odisha,Cuttack,20.46,85.88
Odisha,Bhubaneswar,20.30,85.82
Odisha,Puri,19.81,85.83
Odisha,Sambalpur,21.47,83.97
Odisha,Berhampur,19.32,84.79
Odisha,Rourkela,22.26,84.86
Odisha,Balasore,21.49,86.93
Odisha,Mayurbhanj,21.95,86.72
Odisha,Kendrapara,20.50,86.42
Odisha,Khurda,20.17,85.61
Odisha,Ganjam,19.38,84.70
Odisha,Jajpur,20.85,86.33
Odisha,Koraput,18.81,82.71
Odisha,Bolangir,20.70,83.49
Odisha,Kalahandi,19.91,83.18
Punjab,Ludhiana,30.90,75.85
Punjab,Amritsar,31.63,74.87
Punjab,Jalandhar,31.33,75.57
Punjab,Patiala,30.34,76.39
Punjab,Bathinda,30.21,74.95
Punjab,Sangrur,30.23,75.84
Punjab,Ferozepur,30.93,74.61
Punjab,Hoshiarpur,31.53,75.91
Punjab,Rupnagar,30.97,76.52
Punjab,Fatehgarh Sahib,30.65,76.39
Punjab,Moga,30.82,75.17
Punjab,Muktsar,30.47,74.52
Punjab,Gurudaspur,32.04,75.39
Punjab,Faridkot,30.67,74.76
Punjab,Tarn Taran,31.45,74.93
Rajasthan,Jaipur,26.91,75.79
Rajasthan,Jodhpur,26.30,73.02
Rajasthan,Udaipur,24.58,73.69
Rajasthan,Kota,25.18,75.83
Rajasthan,Bikaner,28.02,73.32
Rajasthan,Alwar,27.55,76.63
Rajasthan,Ajmer,26.45,74.64
Rajasthan,Sikar,27.61,75.14
Rajasthan,Bhilwara,25.35,74.64
Rajasthan,Nagaur,27.20,73.73
Rajasthan,Churu,28.30,74.96
Rajasthan,Bundi,25.44,75.63
Rajasthan,Barmer,25.75,71.40
Rajasthan,Jaisalmer,26.91,70.90
Rajasthan,Hanumangarh,29.58,74.33
Rajasthan,Sriganganagar,29.91,73.88
Rajasthan,Tonk,26.17,75.79
Rajasthan,Dausa,26.89,76.33
Rajasthan,Jhunjhunu,28.13,75.40
Rajasthan,Chhitorgarh,24.88,74.62
Rajasthan,Sawai Madhopur,26.02,76.36
Rajasthan,Dholpur,26.70,77.89
Rajasthan,Zhalawar,24.59,76.16
Tamil Nadu,Chennai,13.08,80.27
Tamil Nadu,Coimbatore,11.02,76.95
Tamil Nadu,Madurai,9.93,78.12
Tamil Nadu,Tiruchirapalli,10.79,78.70
Tamil Nadu,Salem,11.66,78.15
Tamil Nadu,Tirunelveli,8.73,77.70
Tamil Nadu,Thoothukudi,8.80,78.15
Tamil Nadu,Thanjavur,10.79,79.14
Tamil Nadu,Erode,11.34,77.73
Tamil Nadu,Vellore,12.92,79.13
Tamil Nadu,Dindigul,10.37,77.97
Tamil Nadu,Virudhunagar,9.58,77.96
Tamil Nadu,Namakkal,11.22,78.17
Tamil Nadu,Karur,10.96,78.08
Telangana,Hyderabad,17.39,78.49
Telangana,Warangal,17.98,79.59
Telangana,Karimnagar,18.44,79.13
Telangana,Nizamabad,18.67,78.09
Telangana,Khammam,17.25,80.15
Telangana,Mahbubnagar,16.74,77.99
Telangana,Adilabad,19.67,78.53
Telangana,Rangareddy,17.36,78.49
Telangana,Nalgonda,17.05,79.27
Telangana,Medak,18.05,78.26
Uttar Pradesh,Lucknow,26.85,80.95
Uttar Pradesh,Kanpur,26.44,80.33
Uttar Pradesh,Agra,27.18,78.01
Uttar Pradesh,Allahabad,25.43,81.85
Uttar Pradesh,Varanasi,25.32,82.97
Uttar Pradesh,Meerut,28.98,77.71
Uttar Pradesh,Aligarh,27.88,78.08
Uttar Pradesh,Gorakhpur,26.76,83.37
Uttar Pradesh,Moradabad,28.84,78.78
Uttar Pradesh,Bareilly,28.36,79.43
Uttar Pradesh,Saharanpur,29.97,77.55
Uttar Pradesh,Firozabad,27.15,78.40
Uttar Pradesh,Mathura,27.50,77.67
Uttar Pradesh,Bijnor,29.37,78.14
Uttar Pradesh,Etah,27.56,78.66
Uttar Pradesh,Bulandshahr,28.40,77.85
Uttar Pradesh,Muzaffarnagar,29.47,77.70
Uttar Pradesh,Sitapur,27.56,80.68
Uttar Pradesh,Hardoi,27.40,80.13
Uttar Pradesh,Unna,26.56,82.55
Uttar Pradesh,Fatehpur,25.93,80.81
Uttar Pradesh,Pratapgarh,25.89,81.99
Uttar Pradesh,Raebareli,26.23,81.24
Uttar Pradesh,Sultanpur,26.26,82.07
Uttar Pradesh,Azamgarh,26.07,83.18
Uttar Pradesh,Mau,25.94,83.56
Uttar Pradesh,Ghazipur,25.58,83.57
Uttar Pradesh,Ballia,25.75,84.15
Uttar Pradesh,Deoria,26.50,83.78
Uttar Pradesh,CushInagar,26.79,83.90
West Bengal,Kolkata,22.57,88.36
West Bengal,Howrah,22.60,88.26
West Bengal,Bardhaman,23.23,87.86
West Bengal,Nadia,23.47,88.56
West Bengal,Murshidabad,24.18,88.28
West Bengal,Midnapore,22.42,87.32
West Bengal,North 24 Parganas,22.85,88.58
West Bengal,South 24 Parganas,22.15,88.64
West Bengal,Hooghly,22.90,88.39
West Bengal,Malda,25.00,88.14
West Bengal,Jalpaiguri,26.54,88.73
West Bengal,Darjeeling,27.04,88.26
West Bengal,Puruliya,23.33,86.37
West Bengal,Birbhum,23.90,87.54
West Bengal,Bankura,23.23,87.07
West Bengal,North Dinajpur,25.59,88.13
West Bengal,South Dinajpur,25.29,88.75
West Bengal,Cooch Behar,26.32,89.45
"""

os.makedirs('ml/data/raw', exist_ok=True)
with open('ml/data/raw/india_district_coords.csv', 'w') as f:
    f.write(csv_content)
