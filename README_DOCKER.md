# Qlib Production Trading - Docker Deployment Guide

This guide explains how to deploy the Qlib production trading system using Docker.

## 📁 Directory Structure

```
qlib/
├── Dockerfile                    # Main Dockerfile
├── docker-compose.yml            # Multi-service setup
├── .dockerignore                 # Exclude unnecessary files
├── env.example                   # Environment variables template
├── .env                          # Your secrets (create from env.example)
├── docker/
│   └── scripts/
│       ├── build.sh              # Build script
│       ├── run.sh                # Run script
│       └── deploy.sh             # Full deployment script
├── config/
│   └── production_config.yaml    # Production configuration
├── logs/                         # Mounted volume for logs
├── data/                         # Mounted volume for data
└── examples/
    └── production_requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your actual values
nano .env
```

### 2. Deploy with Script

```bash
# Make scripts executable (if not already)
chmod +x docker/scripts/*.sh

# Deploy the system
./docker/scripts/deploy.sh
```

### 3. Check Status

```bash
# View container status
docker ps

# View logs
docker logs -f qlib-trading

# Health check
docker exec qlib-trading python -c "import qlib.production; print('✅ System is healthy')"
```

## 🔧 Manual Deployment

### Build Image

```bash
docker build -t qlib-prod:latest .
```

### Run Container

```bash
# Create directories
mkdir -p logs data config

# Run container
docker run -d \
    --name qlib-trading \
    --restart unless-stopped \
    --env-file .env \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config:/app/config \
    qlib-prod:latest
```

## 🐳 Docker Compose

For a complete setup with monitoring and database:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f qlib-trading

# Stop all services
docker-compose down
```

## 📋 Configuration

### Environment Variables (.env)

```bash
# Broker Configuration
BROKER_TYPE=alpaca  # Options: alpaca, ibkr, mock
PAPER_TRADING=true

# Alpaca Markets
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Data Provider
DATA_PROVIDER=mock  # Options: mock, alpaca, ibkr, polygon
DATA_API_KEY=your_data_api_key

# Risk Management
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=5000
MAX_SECTOR_EXPOSURE=0.3

# Monitoring
ALERT_EMAIL=your_email@example.com
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Database
DB_USERNAME=qlib_user
DB_PASSWORD=qlib_password

# Trading
ENABLE_TRADING=false  # Set to true for live trading
```

### Production Configuration (config/production_config.yaml)

The main configuration file controls:
- Broker settings
- Risk management parameters
- Trading strategy configuration
- Logging and monitoring
- Market hours and emergency procedures

## 📊 Monitoring

### Container Health

```bash
# Check container status
docker ps

# View resource usage
docker stats qlib-trading

# Health check
docker exec qlib-trading python -c "import qlib.production; print('Healthy')"
```

### Logs

```bash
# View real-time logs
docker logs -f qlib-trading

# View recent logs
docker logs --tail 100 qlib-trading

# View logs from host
tail -f logs/production.log
```

### Grafana Dashboard (with docker-compose)

Access Grafana at `http://localhost:3000`
- Username: `admin`
- Password: `admin`

## 🔄 Updates and Maintenance

### Update Code

```bash
# Stop container
docker stop qlib-trading

# Rebuild image
docker build -t qlib-prod:latest .

# Start container
docker start qlib-trading
```

### Update Configuration

```bash
# Edit config file
nano config/production_config.yaml

# Restart container to apply changes
docker restart qlib-trading
```

### Backup Data

```bash
# Backup logs and data
tar -czf backup-$(date +%Y%m%d).tar.gz logs/ data/

# Backup database (if using PostgreSQL)
docker exec qlib-database pg_dump -U qlib_user qlib_trading > backup.sql
```

## 🛠️ Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   # Check logs
   docker logs qlib-trading
   
   # Check if .env file exists
   ls -la .env
   ```

2. **Import errors**
   ```bash
   # Check if dependencies are installed
   docker exec qlib-trading pip list
   
   # Rebuild image
   docker build --no-cache -t qlib-prod:latest .
   ```

3. **Permission issues**
   ```bash
   # Fix directory permissions
   sudo chown -R $USER:$USER logs/ data/
   ```

4. **Network issues (for IBKR)**
   ```bash
   # Check if IBKR TWS/Gateway is accessible
   docker exec qlib-trading ping 127.0.0.1
   ```

### Debug Mode

```bash
# Run container in interactive mode
docker run -it --rm \
    --env-file .env \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config:/app/config \
    qlib-prod:latest bash

# Inside container
python -m qlib.production.workflow --config config/production_config.yaml
```

## 🔒 Security Best Practices

1. **Never commit secrets**
   - Keep `.env` file out of version control
   - Use Docker secrets in production

2. **Network security**
   - Don't expose trading container to public internet
   - Use VPN for remote access

3. **Regular updates**
   - Keep base image updated
   - Regularly update dependencies

4. **Backup strategy**
   - Regular backups of logs and data
   - Test restore procedures

## 📈 Production Deployment

### Cloud Providers

- **AWS**: Use EC2 with ECS or EKS
- **GCP**: Use Compute Engine with GKE
- **Azure**: Use VM with AKS
- **DigitalOcean**: Use Droplets with Docker

### High Availability

- Use multiple containers behind a load balancer
- Implement health checks and auto-restart
- Use persistent volumes for data storage

### Monitoring Stack

- **Logs**: ELK stack or CloudWatch
- **Metrics**: Prometheus + Grafana
- **Alerts**: PagerDuty or Slack
- **APM**: New Relic or DataDog

## 📞 Support

For issues and questions:
1. Check the logs first
2. Review this documentation
3. Check the main Qlib documentation
4. Open an issue on GitHub

---

**Happy Trading! 🚀** 