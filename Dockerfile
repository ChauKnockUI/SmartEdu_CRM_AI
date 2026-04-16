# Sử dụng Python 3.10 slim để image nhẹ nhưng vẫn cài được các thư viện Data Science
FROM python:3.10-slim

# Thiết lập thư mục làm việc (Working Directory)
WORKDIR /app

# Cài đặt một số thư viện build cơ bản (cần thiết cho vài package Python)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào trước để tận dụng Docker Cache
COPY requirements.txt .

# Do dự án sử dụng thêm SHAP và XGBoost (không thấy có sẵn trong thư mục gốc)
# Nên ta sẽ cài thủ công thêm 2 thư viện này để đảm bảo Explainable AI hoạt động
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir shap xgboost

# Copy toàn bộ source code vào container
COPY . .

# Expose port (tương ứng với lúc chạy Uvicorn ở localhost)
EXPOSE 8000

# Lệnh khởi chạy mặc định khi container run
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
