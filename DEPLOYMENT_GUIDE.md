# Deployment Guide for ReadWave Quant Trading System

## Overview
This guide covers deploying the automated trading system to cloud services like AWS EC2, Google Cloud, or any Docker-compatible platform.

## Prerequisites

### 1. Environment Variables
Create a `.env` file in the project root:
```bash
# Alpaca Trading API (Paper Trading)
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
ALPACA_PAPER_TRADING=true

# Alpaca Trading API (Live Trading - USE WITH CAUTION)
# ALPACA_API_KEY=your_live_api_key_here
# ALPACA_SECRET_KEY=your_live_secret_key_here
# ALPACA_PAPER_TRADING=false

# Optional: Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=trading_user
DB_PASSWORD=secure_password
```

### 2. Configuration Files
Ensure your configuration files are properly set up:
- `config/alpaca_config.yaml` - Alpaca broker configuration
- `config/live_trading_config.yaml` - Live trading parameters

## Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Local Docker Testing
```bash
# Build the Docker image
docker build -t readwave-trading .

# Run with environment variables
docker run --env-file .env -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data readwave-trading

# Or use docker-compose
docker-compose up -d
```

#### Cloud Deployment (AWS EC2)

1. **Launch EC2 Instance**
   ```bash
   # Ubuntu 20.04 LTS recommended
   # t3.medium or larger for production
   # At least 20GB storage
   ```

2. **Install Docker on EC2**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

3. **Deploy Application**
   ```bash
   # Clone your repository
   git clone <your-repo-url>
   cd readwave_quant
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your actual API keys
   
   # Build and run
   docker-compose up -d
   ```

4. **Monitor Deployment**
   ```bash
   # Check container status
   docker-compose ps
   
   # View logs
   docker-compose logs -f trading-system
   
   # Check health
   docker-compose exec trading-system python3 -c "import sys; print('System healthy')"
   ```

### Option 2: Direct Python Deployment

#### EC2 Setup
```bash
# Install Python 3.9+
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-pip python3.9-venv

# Create virtual environment
python3.9 -m venv trading_env
source trading_env/bin/activate

# Install dependencies
pip install -r requirements-docker.txt

# Set environment variables
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
export ALPACA_PAPER_TRADING=true

# Run the system
python3 -m src.trading.live_trading --auto-select --max-stocks 10
```

## Production Considerations

### 1. Security
- Use IAM roles instead of hardcoded credentials
- Enable VPC and security groups
- Use HTTPS for all API communications
- Regularly rotate API keys

### 2. Monitoring
```bash
# Set up CloudWatch monitoring
# Monitor CPU, memory, disk usage
# Set up alerts for system failures

# Log monitoring
docker-compose logs -f --tail=100 trading-system
```

### 3. Backup Strategy
- Backup configuration files
- Backup trading logs
- Consider database backup if using persistent storage

### 4. Scaling
- Use auto-scaling groups for high availability
- Consider load balancers for multiple instances
- Use ECS/EKS for container orchestration

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Check API keys
   echo $ALPACA_API_KEY
   echo $ALPACA_SECRET_KEY
   
   # Test connection
   python3 -c "import alpaca_trade_api as api; print('API connection test')"
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

3. **Timezone Issues**
   ```bash
   # Set timezone in Dockerfile
   ENV TZ=America/New_York
   ```

### Log Analysis
```bash
# View recent logs
tail -f logs/live_trading.log

# Search for errors
grep -i error logs/live_trading.log

# Monitor trading activity
grep -i "trade\|order" logs/live_trading.log
```

## Performance Optimization

### 1. Resource Allocation
- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ for production
- **Storage**: SSD recommended for data processing

### 2. Network Optimization
- Use VPC endpoints for AWS services
- Optimize API call frequency
- Implement connection pooling

### 3. Caching
- Cache frequently accessed data
- Use Redis for session management
- Implement data caching strategies

## Emergency Procedures

### 1. Emergency Stop
```bash
# Stop all trading immediately
docker-compose exec trading-system python3 -m src.trading.live_trading --emergency-stop

# Or stop the entire system
docker-compose down
```

### 2. Data Recovery
```bash
# Backup current state
docker-compose exec trading-system tar -czf backup-$(date +%Y%m%d).tar.gz logs/ data/

# Restore from backup
docker-compose exec trading-system tar -xzf backup-20231201.tar.gz
```

## Maintenance

### 1. Regular Updates
```bash
# Update code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 2. Health Checks
```bash
# Automated health check
curl -f http://localhost:8080/health || exit 1

# Manual health check
docker-compose exec trading-system python3 -c "import sys; sys.exit(0)"
```

## Support

For issues and questions:
1. Check the logs first
2. Review this deployment guide
3. Check the main README.md
4. Open an issue in the repository

---

**⚠️ Important Notes:**
- Always test with paper trading first
- Monitor the system continuously
- Keep backups of all configurations
- Document any customizations
- Test emergency procedures regularly 