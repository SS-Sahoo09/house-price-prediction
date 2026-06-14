# =========================================
# app.py
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from math import pow

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="House Price Prediction and Market Analytics Platform",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(open("xgboost_house_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#111827;
}

.sub-title{
    font-size:18px;
    color:#6B7280;
    margin-bottom:25px;
}

.section{
    background:#F9FAFB;
    padding:24px;
    border-radius:16px;
    margin-bottom:20px;
    border:1px solid #E5E7EB;
}

.section-title{
    font-size:24px;
    font-weight:600;
    margin-bottom:18px;
    color:#111827;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Investment Analysis",
        "EMI Calculator",
        "Market Dashboard"
    ]
)

# =========================================
# TITLE
# =========================================

st.markdown(
    '<div class="main-title">AI Real Estate Analytics Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Advanced Property Valuation and Investment Intelligence System</div>',
    unsafe_allow_html=True
)

# =========================================
# STATE CITY MAP
# =========================================

state_city_map = {

    "Andhra Pradesh": [
        "Visakhapatnam",
        "Vijayawada",
        "Guntur",
        "Nellore",
        "Kurnool",
        "Tirupati",
        "Rajahmundry"
    ],

    "Arunachal Pradesh": [
        "Itanagar",
        "Tawang",
        "Pasighat",
        "Ziro",
        "Bomdila",
        "Roing",
        "Naharlagun"
    ],

    "Assam": [
        "Guwahati",
        "Silchar",
        "Dibrugarh",
        "Jorhat",
        "Tezpur",
        "Nagaon",
        "Tinsukia"
    ],

    "Bihar": [
        "Patna",
        "Gaya",
        "Muzaffarpur",
        "Bhagalpur",
        "Darbhanga",
        "Purnia",
        "Ara"
    ],

    "Chhattisgarh": [
        "Raipur",
        "Bilaspur",
        "Durg",
        "Korba",
        "Jagdalpur",
        "Raigarh",
        "Ambikapur"
    ],

    "Goa": [
        "Panaji",
        "Margao",
        "Vasco da Gama",
        "Mapusa",
        "Ponda",
        "Bicholim",
        "Canacona"
    ],

    "Gujarat": [
        "Ahmedabad",
        "Surat",
        "Vadodara",
        "Rajkot",
        "Bhavnagar",
        "Jamnagar",
        "Gandhinagar"
    ],

    "Haryana": [
        "Gurgaon",
        "Faridabad",
        "Panipat",
        "Ambala",
        "Hisar",
        "Rohtak",
        "Karnal"
    ],

    "Himachal Pradesh": [
        "Shimla",
        "Manali",
        "Dharamshala",
        "Solan",
        "Kullu",
        "Mandi",
        "Hamirpur"
    ],

    "Jharkhand": [
        "Ranchi",
        "Jamshedpur",
        "Dhanbad",
        "Bokaro",
        "Deoghar",
        "Hazaribagh",
        "Giridih"
    ],

    "Karnataka": [
        "Bangalore",
        "Mysore",
        "Mangalore",
        "Hubli",
        "Belgaum",
        "Shimoga",
        "Davanagere"
    ],

    "Kerala": [
        "Kochi",
        "Trivandrum",
        "Kozhikode",
        "Thrissur",
        "Kannur",
        "Alappuzha",
        "Palakkad"
    ],

    "Madhya Pradesh": [
        "Indore",
        "Bhopal",
        "Gwalior",
        "Jabalpur",
        "Ujjain",
        "Sagar",
        "Satna"
    ],

    "Maharashtra": [
        "Mumbai",
        "Pune",
        "Nagpur",
        "Nashik",
        "Aurangabad",
        "Kolhapur",
        "Solapur"
    ],

    "Manipur": [
        "Imphal",
        "Thoubal",
        "Bishnupur",
        "Churachandpur",
        "Ukhrul",
        "Kakching",
        "Senapati"
    ],

    "Meghalaya": [
        "Shillong",
        "Tura",
        "Jowai",
        "Nongpoh",
        "Baghmara",
        "Williamnagar",
        "Resubelpara"
    ],

    "Mizoram": [
        "Aizawl",
        "Lunglei",
        "Champhai",
        "Kolasib",
        "Serchhip",
        "Saiha",
        "Mamit"
    ],

    "Nagaland": [
        "Kohima",
        "Dimapur",
        "Mokokchung",
        "Tuensang",
        "Wokha",
        "Zunheboto",
        "Phek"
    ],

    "Odisha": [
        "Bhubaneswar",
        "Cuttack",
        "Rourkela",
        "Puri",
        "Sambalpur",
        "Berhampur",
        "Balasore"
    ],

    "Punjab": [
        "Ludhiana",
        "Amritsar",
        "Jalandhar",
        "Patiala",
        "Bathinda",
        "Mohali",
        "Pathankot"
    ],

    "Rajasthan": [
        "Jaipur",
        "Udaipur",
        "Jodhpur",
        "Kota",
        "Ajmer",
        "Bikaner",
        "Alwar"
    ],

    "Sikkim": [
        "Gangtok",
        "Namchi",
        "Gyalshing",
        "Mangan",
        "Rangpo",
        "Singtam",
        "Jorethang"
    ],

    "Tamil Nadu": [
        "Chennai",
        "Coimbatore",
        "Madurai",
        "Salem",
        "Tiruchirappalli",
        "Erode",
        "Vellore"
    ],

    "Telangana": [
        "Hyderabad",
        "Warangal",
        "Karimnagar",
        "Nizamabad",
        "Khammam",
        "Mahbubnagar",
        "Adilabad"
    ],

    "Tripura": [
        "Agartala",
        "Udaipur",
        "Dharmanagar",
        "Kailasahar",
        "Belonia",
        "Ambassa",
        "Sonamura"
    ],

    "Uttar Pradesh": [
        "Lucknow",
        "Noida",
        "Kanpur",
        "Varanasi",
        "Agra",
        "Prayagraj",
        "Ghaziabad"
    ],

    "Uttarakhand": [
        "Dehradun",
        "Haridwar",
        "Nainital",
        "Haldwani",
        "Rishikesh",
        "Roorkee",
        "Mussoorie"
    ],

    "West Bengal": [
        "Kolkata",
        "Siliguri",
        "Durgapur",
        "Asansol",
        "Howrah",
        "Kharagpur",
        "Darjeeling"
    ],

    "Delhi": [
        "New Delhi",
        "Dwarka",
        "Rohini",
        "Saket",
        "Karol Bagh",
        "Janakpuri",
        "Pitampura"
    ],

    "Jammu and Kashmir": [
        "Srinagar",
        "Jammu",
        "Anantnag",
        "Baramulla",
        "Kupwara",
        "Pulwama",
        "Kathua"
    ],

    "Ladakh": [
        "Leh",
        "Kargil",
        "Nubra",
        "Diskit",
        "Zanskar",
        "Drass",
        "Padum"
    ]
}

# =========================================
# COMMON PROPERTY INPUTS
# =========================================

if page in [
    "Home",
    "Investment Analysis"
]:

    # =====================================
    # BASIC PROPERTY DETAILS
    # =====================================

    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Basic Property Details</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        area_sqft = st.number_input(
            "Area (sqft)",
            400,
            10000,
            1500
        )

        bhk = st.number_input(
            "BHK",
            1,
            10,
            3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            1,
            10,
            2
        )

    with col2:

        floors = st.number_input(
            "Floors",
            1,
            50,
            2
        )

        property_type = st.selectbox(
            "Property Type",
            [
                "Apartment",
                "Villa",
                "Independent House",
                "Studio"
            ]
        )

        age = st.number_input(
            "Property Age",
            0,
            100,
            5
        )

    with col3:

        garden_area = st.number_input(
            "Garden Area",
            0,
            5000,
            200
        )

        garage_capacity = st.number_input(
            "Garage Capacity",
            0,
            10,
            1
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================
    # LOCATION INFORMATION
    # =====================================

    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Location Information</div>',
        unsafe_allow_html=True
    )

    col4, col5, col6 = st.columns(3)

    with col4:

        state = st.selectbox(
            "State",
            list(state_city_map.keys())
        )

        locality_type = st.selectbox(
            "Locality Type",
            [
                "Luxury",
                "Premium",
                "Urban",
                "Suburban",
                "Developing",
                "Affordable"
            ]
        )

    with col5:

        city = st.selectbox(
            "City",
            state_city_map[state]
        )

        crime_rate = st.slider(
            "Crime Rate",
            1.0,
            10.0,
            3.0
        )

    with col6:

        traffic_score = st.slider(
            "Traffic Score",
            1.0,
            10.0,
            4.0
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================
    # AMENITIES
    # =====================================

    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Amenities</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        school = st.checkbox("Nearby School")

    with c2:
        hospital = st.checkbox("Nearby Hospital")

    with c3:
        mall = st.checkbox("Nearby Mall")

    st.write("")

    d1, d2, d3 = st.columns(3)

    with d1:

        water_supply = st.selectbox(
            "Water Supply",
            ["24x7", "Borewell", "Municipal"]
        )

    with d2:

        electricity_backup = st.selectbox(
            "Electricity Backup",
            ["Yes", "No"]
        )

    with d3:

        facing = st.selectbox(
            "Facing",
            ["North", "South", "East", "West"]
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================
    # MARKET INTELLIGENCE
    # =====================================

    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Market Intelligence</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        demand_index = st.slider(
            "Demand Score",
            1.0,
            10.0,
            8.0
        )

        builder_reputation = st.slider(
            "Builder Reputation",
            1.0,
            10.0,
            8.0
        )

    with m2:

        infrastructure_score = st.slider(
            "Infrastructure Score",
            1.0,
            10.0,
            8.0
        )

        future_development_score = st.slider(
            "Future Development Score",
            1.0,
            10.0,
            8.0
        )

    with m3:

        road_connectivity = st.slider(
            "Road Connectivity",
            1,
            10,
            7
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================
    # MODEL INPUT
    # =====================================

    input_data = {

        'state': state,
        'district_city': city,
        'locality_type': locality_type,
        'area_sqft': area_sqft,
        'bhk': bhk,
        'bathrooms': bathrooms,
        'floors': floors,
        'garden_area': garden_area,
        'garage_capacity': garage_capacity,
        'age': age,
        'property_type': property_type,
        'furnishing': "Semi-Furnished",
        'school': int(school),
        'hospital': int(hospital),
        'mall': int(mall),
        'road_connectivity': road_connectivity,
        'facing': facing,
        'water_supply': water_supply,
        'electricity_backup': electricity_backup,
        'crime_rate': crime_rate,
        'traffic_score': traffic_score,
        'demand_index': demand_index,
        'builder_reputation': builder_reputation,
        'future_development_score': future_development_score,
        'infrastructure_score': infrastructure_score,
        'nearby_metro_km': 5
    }

    input_df = pd.DataFrame([input_data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]

# =========================================
# HOME PAGE
# =========================================

if page == "Home":

    if st.button("Predict Price"):

        # =====================================
        # CATEGORY
        # =====================================

        if prediction < 8000000:
            category = "Budget Housing"
        
        elif prediction < 15000000:
            category = "Affordable Residential"
        
        elif prediction < 30000000:
            category = "Mid-Range Property"
        
        elif prediction < 50000000:
            category = "Premium Property"
        
        elif prediction < 80000000:
            category = "Luxury Residence"
        
        elif prediction < 150000000:
            category = "Ultra Luxury Property"
        
        elif prediction < 300000000:
            category = "Elite Investment Estate"
        
        else:
            category = "Iconic Luxury Landmark"

        # =====================================
        # PREDICTION RESULT
        # =====================================

        st.subheader("Estimated Property Price")

        st.success(f"₹ {prediction:,.0f}")

        st.write(f"### Category: {category}")

        # =====================================
        # METRICS
        # =====================================

        investment_score = (
            demand_index * 0.30 +
            infrastructure_score * 0.30 +
            builder_reputation * 0.20 +
            future_development_score * 0.20
        ) * 10

        appreciation = (
            demand_index +
            infrastructure_score +
            future_development_score
        ) / 3

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Investment Score",
                f"{investment_score:.1f}/100"
            )

        with col2:
            st.metric(
                "Demand Score",
                f"{demand_index}/10"
            )

        with col3:
            st.metric(
                "Expected Appreciation",
                f"{appreciation:.1f}/10"
            )

        # =====================================
        # SMART INSIGHT
        # =====================================

        # =====================================
        # SMART INSIGHTS
        # =====================================
        
        if (
            demand_index >= 9 and
            infrastructure_score >= 9
        ):
        
            st.success(
                "Exceptional market demand and infrastructure indicate very strong appreciation potential."
            )
        
        elif (
            demand_index >= 8
        ):
        
            st.success(
                "High buyer demand suggests strong resale opportunities and rental potential."
            )
        
        elif (
            demand_index <= 4
        ):
        
            st.warning(
                "Lower market demand could affect resale speed and long-term investment returns."
            )
        
        elif (
            builder_reputation >= 9
        ):
        
            st.success(
                "Top-tier builder reputation significantly improves trust, resale value, and construction reliability."
            )
        
        elif (
            builder_reputation <= 4
        ):
        
            st.warning(
                "Lower builder reputation may impact future buyer confidence and resale demand."
            )
        
        elif (
            infrastructure_score >= 9
        ):
        
            st.success(
                "Excellent infrastructure and urban connectivity make this location highly attractive for long-term investment."
            )
        
        elif (
            infrastructure_score <= 4
        ):
        
            st.warning(
                "Weak infrastructure development may limit future appreciation potential."
            )
        
        elif (
            future_development_score >= 9
        ):
        
            st.success(
                "Upcoming development projects may significantly increase future property valuation."
            )
        
        elif (
            future_development_score <= 4
        ):
        
            st.warning(
                "Limited future development may slow long-term appreciation growth."
            )
        
        elif (
            crime_rate >= 8
        ):
        
            st.warning(
                "High crime rate may negatively impact livability and investment attractiveness."
            )
        
        elif (
            traffic_score >= 8
        ):
        
            st.warning(
                "Heavy traffic congestion could reduce long-term residential demand."
            )
        
        elif (
            locality_type == "Luxury"
        ):
        
            st.info(
                "Luxury localities generally maintain stronger appreciation and premium rental demand."
            )
        
        elif (
            road_connectivity >= 8
        ):
        
            st.success(
                "Excellent road connectivity improves accessibility and investment attractiveness."
            )
        
        elif (
            school and hospital and mall
        ):
        
            st.success(
                "Strong nearby amenities improve family suitability and rental appeal."
            )
        
        elif (
            investment_score >= 85
        ):
        
            st.success(
                "Overall investment indicators suggest this property has strong growth and appreciation potential."
            )
        
        else:
        
            st.info(
                "This property shows balanced characteristics with moderate investment potential."
            )

        # =====================================
        # FEATURE IMPORTANCE GRAPH
        # =====================================

        feature_importance = pd.DataFrame({

            "Feature": [
                "Demand",
                "Infrastructure",
                "Builder Reputation",
                "Future Development",
                "Safety",
                "Traffic"
            ],

            "Importance": [
                demand_index,
                infrastructure_score,
                builder_reputation,
                future_development_score,
                10 - crime_rate,
                10 - traffic_score
            ]
        })

        fig = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            color="Importance",
            title="Feature Importance Analysis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =========================================
# INVESTMENT ANALYSIS
# =========================================

if page == "Investment Analysis":

    st.markdown("## Investment Analysis")

    risk_score = (
        crime_rate * 0.4 +
        traffic_score * 0.3 +
        (10 - demand_index) * 0.3
    )

    if risk_score < 4:
        risk = "Low Risk"

    elif risk_score < 7:
        risk = "Moderate Risk"

    else:
        risk = "High Risk"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Level", risk)

    with col2:
        st.metric(
            "Current Price",
            f"₹ {prediction:,.0f}"
        )

    with col3:
        st.metric(
            "Expected ROI",
            f"{future_development_score * 4:.1f}%"
        )

    # =====================================
    # FUTURE PRICE FORECAST
    # =====================================

    years = [2025, 2026, 2027, 2028, 2029]

    prices = [
        prediction,
        prediction * 1.08,
        prediction * 1.16,
        prediction * 1.24,
        prediction * 1.35
    ]

    forecast_df = pd.DataFrame({

        "Year": years,
        "Price": prices
    })

    fig1 = px.line(
        forecast_df,
        x="Year",
        y="Price",
        markers=True,
        title="Future Property Price Forecast"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================
    # RADAR CHART
    # =====================================

    radar_fig = go.Figure()

    radar_fig.add_trace(go.Scatterpolar(

        r=[
            demand_index,
            infrastructure_score,
            10 - crime_rate,
            10 - traffic_score,
            road_connectivity
        ],

        theta=[
            "Demand",
            "Infrastructure",
            "Safety",
            "Traffic",
            "Connectivity"
        ],

        fill='toself'
    ))

    radar_fig.update_layout(

        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,10]
            )
        ),

        title="Property Score Radar Chart"
    )

    st.plotly_chart(
        radar_fig,
        use_container_width=True
    )

# =========================================
# EMI CALCULATOR
# =========================================

if page == "EMI Calculator":

    st.markdown("## EMI Calculator")

    property_price = st.number_input(
        "Property Price",
        min_value=100000,
        value=5000000,
        step=100000
    )

    down_payment = st.number_input(
        "Down Payment",
        min_value=0,
        value=1000000,
        step=50000
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        max_value=20.0,
        value=8.5
    )

    years = st.number_input(
        "Loan Tenure (Years)",
        min_value=1,
        max_value=40,
        value=20
    )

    principal = property_price - down_payment

    monthly_rate = interest_rate / 12 / 100

    months = years * 12

    emi = (
        principal *
        monthly_rate *
        pow(1 + monthly_rate, months)
    ) / (
        pow(1 + monthly_rate, months) - 1
    )

    total_payment = emi * months

    total_interest = total_payment - principal

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Monthly EMI",
            f"₹ {emi:,.0f}"
        )

    with col2:
        st.metric(
            "Total Interest",
            f"₹ {total_interest:,.0f}"
        )

    with col3:
        st.metric(
            "Total Payment",
            f"₹ {total_payment:,.0f}"
        )

# =========================================
# MARKET DASHBOARD
# =========================================

if page == "Market Dashboard":

    st.markdown("## Real Estate Market Dashboard")

    # =====================================
    # CITY PRICE COMPARISON
    # =====================================

    city_df = pd.DataFrame({

        "City": [
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Hyderabad",
            "Pune",
            "Chennai"
        ],

        "Average Price": [
            9.5,
            8.9,
            8.2,
            6.5,
            5.9,
            5.2
        ]
    })

    fig1 = px.bar(
        city_df,
        x="City",
        y="Average Price",
        color="Average Price",
        title="City-wise Average Property Prices"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================
    # DEMAND HEATMAP
    # =====================================

    heatmap_df = pd.DataFrame(
        np.random.rand(6,6),
        columns=[
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Pune",
            "Hyderabad",
            "Chennai"
        ]
    )

    fig2 = px.imshow(
        heatmap_df,
        text_auto=True,
        title="Demand Heatmap"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =====================================
    # ROI COMPARISON
    # =====================================

    roi_df = pd.DataFrame({

        "City": [
            "Hyderabad",
            "Pune",
            "Bangalore",
            "Delhi",
            "Mumbai"
        ],

        "ROI": [
            28,
            24,
            22,
            18,
            16
        ]
    })

    fig3 = px.bar(
        roi_df,
        x="ROI",
        y="City",
        orientation='h',
        color="ROI",
        title="ROI Comparison by City"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =====================================
    # PROPERTY PRICE DISTRIBUTION
    # =====================================

    distribution_df = pd.DataFrame({

        "Price": np.random.normal(
            50000000,
            15000000,
            1000
        )
    })

    fig4 = px.histogram(
        distribution_df,
        x="Price",
        nbins=40,
        title="Property Price Distribution"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # =====================================
    # DEMAND VS PRICE
    # =====================================

    scatter_df = pd.DataFrame({

        "Demand": [5,6,7,8,9,10],
        "Price": [3,4,5,6,8,10]
    })

    fig5 = px.scatter(
        scatter_df,
        x="Demand",
        y="Price",
        size="Price",
        title="Demand vs Property Price"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # =====================================
    # INFRASTRUCTURE VS APPRECIATION
    # =====================================

    infra_df = pd.DataFrame({

        "Infrastructure": [4,5,6,7,8,9],
        "Appreciation": [5,6,7,8,9,10]
    })

    fig6 = px.scatter(
        infra_df,
        x="Infrastructure",
        y="Appreciation",
        size="Appreciation",
        title="Infrastructure vs Appreciation"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    # =====================================
    # LUXURY CITY RANKING
    # =====================================

    luxury_df = pd.DataFrame({

        "City": [
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Pune",
            "Hyderabad"
        ],

        "Luxury Index": [
            95,
            90,
            88,
            75,
            70
        ]
    })

    fig7 = px.line(
        luxury_df,
        x="City",
        y="Luxury Index",
        markers=True,
        title="Luxury City Ranking"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

    # =====================================
    # PROPERTY TYPE DISTRIBUTION
    # =====================================

    property_df = pd.DataFrame({

        "Type": [
            "Apartment",
            "Villa",
            "Independent House",
            "Studio"
        ],

        "Count": [
            45,
            25,
            20,
            10
        ]
    })

    fig8 = px.pie(
        property_df,
        names="Type",
        values="Count",
        title="Property Type Distribution"
    )

    st.plotly_chart(
        fig8,
        use_container_width=True
    )