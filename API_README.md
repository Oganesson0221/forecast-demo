# Bank Balance Forecasting API

A Flask-based REST API for forecasting bank account balances using XGBoost. Designed for easy deployment on Render.

## 🚀 Quick Deploy to Render

### Option 1: Using render.yaml (Blueprint)

1. Push this repository to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click **New** → **Blueprint**
4. Connect your GitHub repository
5. Render will automatically detect `render.yaml` and deploy

### Option 2: Manual Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `forecast-api` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-api.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Click **Create Web Service**

## 🔧 Environment Variables

Set these in Render's Environment tab:

| Variable | Required | Default | Description                                             |
| -------- | -------- | ------- | ------------------------------------------------------- |
| `PORT`   | No       | `5000`  | Port to run the server (Render sets this automatically) |
| `DEBUG`  | No       | `false` | Enable debug mode (`true`/`false`)                      |

**No additional environment variables are required for basic functionality.**

## 📡 API Endpoints

### Health Check

```
GET /
GET /health
```

### Train Model

```
POST /train
Content-Type: application/json

{
    "data": "Date\tDescription\tName\tCategory\tCredit\tDebit\tBalance\tCurrency\n10/27/2025\tPayment\tAmazon\tMarketplace\t4,685.33\t\t34,587.86\tUSD\n..."
}
```

Or with JSON array:

```json
{
  "transactions": [
    {
      "date": "2025-10-27",
      "description": "Payment from Amazon",
      "name": "Amazon",
      "category": "Marketplace",
      "credit": 4685.33,
      "debit": 0,
      "balance": 34587.86
    }
  ]
}
```

### Forecast Future Days

```
POST /forecast
Content-Type: application/json

{
    "days": 7
}
```

### Train and Forecast (One-Step)

```
POST /train-and-forecast
Content-Type: application/json

{
    "data": "...(transaction data)...",
    "forecast_days": 14
}
```

## 📊 Data Format

Your transaction data should have these columns:

| Column      | Type   | Description                                                |
| ----------- | ------ | ---------------------------------------------------------- |
| Date        | String | Transaction date (MM/DD/YYYY or YYYY-MM-DD)                |
| Description | String | Transaction description                                    |
| Name        | String | Payee/Payer name                                           |
| Category    | String | Transaction category                                       |
| Credit      | Number | Credit amount (can be empty)                               |
| Debit       | Number | Debit amount (can be empty, use parentheses for negatives) |
| Balance     | Number | Account balance after transaction                          |
| Currency    | String | Currency code (optional)                                   |

### Example Data (Tab-Separated)

```
Date	Description	Name	Category	Credit	Debit	Balance	Currency
10/27/2025	Real Time Payment From Amazon.com	Amazon	Marketplace Payments	4,685.33		34,587.86	USD
10/27/2025	Electronic Deposit	AfterPay	ENTERTAINMENT	2,258.80		29,902.53	USD
10/27/2025	Electronic Deposit	Shopify	Marketplace Payments	4,204.92		27,643.73	USD
10/27/2025	Internet Banking Payment	Credit Line	LOAN_PAYMENTS		(3,264.65)	17,683.80	USD
```

## 📋 Example Usage

### Python

```python
import requests

# Your Render URL
API_URL = "https://your-app.onrender.com"

# Prepare your transaction data
data = """Date	Description	Name	Category	Credit	Debit	Balance	Currency
10/27/2025	Payment	Amazon	Marketplace	4,685.33		34,587.86	USD
10/26/2025	Deposit	Shopify	Marketplace	2,000.00		29,902.53	USD
...(more data)..."""

# Train and forecast
response = requests.post(
    f"{API_URL}/train-and-forecast",
    json={
        "data": data,
        "forecast_days": 7
    }
)

result = response.json()
print(result['forecast']['predictions'])
```

### cURL

```bash
# Health check
curl https://your-app.onrender.com/health

# Train and forecast
curl -X POST https://your-app.onrender.com/train-and-forecast \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...], "forecast_days": 7}'
```

### JavaScript/Fetch

```javascript
const API_URL = "https://your-app.onrender.com";

const response = await fetch(`${API_URL}/train-and-forecast`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    transactions: [
      { date: "2025-10-27", credit: 4685.33, debit: 0, balance: 34587.86 },
      // ... more transactions
    ],
    forecast_days: 7,
  }),
});

const data = await response.json();
console.log(data.forecast.predictions);
```

## 📈 Response Examples

### Train Response

```json
{
  "success": true,
  "message": "Model trained successfully",
  "data_points": 180,
  "training_samples": 120,
  "test_samples": 30,
  "metrics": {
    "MAE": 1234.56,
    "RMSE": 2345.67,
    "MAPE": 3.45
  },
  "date_range": {
    "start": "2025-04-01",
    "end": "2025-10-27"
  }
}
```

### Forecast Response

```json
{
  "success": true,
  "forecast_days": 7,
  "last_known_date": "2025-10-27",
  "last_known_balance": 34587.86,
  "forecasts": [
    { "date": "2025-10-28", "predicted_balance": 35000.12 },
    { "date": "2025-10-29", "predicted_balance": 35500.45 },
    { "date": "2025-10-30", "predicted_balance": 36100.78 }
  ]
}
```

## 🔒 Notes

- The model is trained in-memory and resets when the server restarts
- For production use, consider adding model persistence (save/load to file or database)
- Minimum 30 days of data required for training
- Maximum forecast horizon is 365 days

## 🛠 Local Development

```bash
# Install dependencies
pip install -r requirements-api.txt

# Run locally
python app.py

# Or with gunicorn
gunicorn app:app --bind 0.0.0.0:5000
```

## 📝 License

MIT License
