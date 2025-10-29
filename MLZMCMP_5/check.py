import requests

url = 'http://127.0.0.1:9696/predict'

customer = {
    "lead_source": "events",
    "industry": "other",
    "number_of_courses_viewed": 9,
    "annual_income": 20,
    "employment_status": "NA",
    "location": "asia",
    "interaction_count": 3,
    "lead_score": 0.2
}

ans = requests.post(url, json = customer).json()

print(ans)

if ans['converted']:
        print("this customer will convert")
else:
    print("this customer will not convert")

