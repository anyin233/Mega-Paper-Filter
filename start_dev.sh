#!/bin/bash
# Start both backend and frontend servers for development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Paper Labeler Development Servers${NC}"
echo "=============================================="

# Check if backend dependencies are installed
echo -e "${YELLOW}📦 Checking backend dependencies...${NC}"
if ! uv --version > /dev/null 2>&1; then
    echo -e "${RED}❌ uv not found. Please install uv first.${NC}"
    exit 1
fi

# Install backend dependencies
echo -e "${YELLOW}📦 Installing backend dependencies...${NC}"
uv sync

# Check if frontend dependencies are installed
echo -e "${YELLOW}📦 Checking frontend dependencies...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd frontend && npm install && cd ..
fi

# Start backend in background
echo -e "${YELLOW}🔧 Starting backend server on port 8000...${NC}"
uv run python start_backend.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Test backend health
echo -e "${YELLOW}🏥 Testing backend health...${NC}"
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo -e "${YELLOW}🌐 Starting frontend development server on port 3000...${NC}"
cd frontend && npm start &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}🎉 Both servers are starting up!${NC}"
echo "=============================================="
echo -e "${GREEN}Frontend:${NC} http://localhost:3000"
echo -e "${GREEN}Backend API:${NC} http://localhost:8000"
echo -e "${GREEN}API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT

# Wait for either process to exit
wait