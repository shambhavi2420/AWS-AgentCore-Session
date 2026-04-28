import os
from bedrock_agentcore.runtime import (
    BedrockAgentCoreApp,
)  #### AGENTCORE RUNTIME - LINE 1 ####
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
import requests
import boto3
from strands.models import BedrockModel
from lab_helpers.utils import get_ssm_parameter
from lab_helpers.lab1_strands_agent import (
    get_return_policy,
    get_product_info,
    get_technical_support,
    SYSTEM_PROMPT,
    MODEL_ID,
)

from lab_helpers.lab2_memory import (
    ACTOR_ID,
    SESSION_ID,
)
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

# Initialize boto3 client
sts_client = boto3.client('sts')

# Get AWS account details
REGION = boto3.session.Session().region_name

# Lab1 import: Create the Bedrock model
model = BedrockModel(model_id=MODEL_ID)

# Lab2 import: Memory
memory_id = os.environ.get("MEMORY_ID")
if not memory_id:
    raise Exception("Environment variable MEMORY_ID is required")

# Initialize the AgentCore Runtime App
app = BedrockAgentCoreApp()  #### AGENTCORE RUNTIME - LINE 2 ####

@app.entrypoint  #### AGENTCORE RUNTIME - LINE 3 ####
async def invoke(payload, context=None):
    """AgentCore Runtime entrypoint function"""
    user_input = payload.get("prompt", "")
    session_id = context.session_id # Get session_id from context
    actor_id = payload.get("actor_id", ACTOR_ID) 
    # Access request headers - handle None case
    request_headers = context.request_headers or {}

    # Get Client JWT token
    auth_header = request_headers.get('Authorization', '')

    print(f"Authorization header: {auth_header}")
    # Get Gateway ID
    existing_gateway_id = get_ssm_parameter("/app/customersupport/agentcore/gateway_id")
    
    # Initialize Bedrock AgentCore Control client
    gateway_client = boto3.client(
        "bedrock-agentcore-control",
        region_name=REGION,
    )
    # Get existing gateway details
    gateway_response = gateway_client.get_gateway(gatewayIdentifier=existing_gateway_id)

    # Get gateway url
    gateway_url = gateway_response['gatewayUrl']

    # Create MCP client and agent within context manager if JWT token available
    if gateway_url and auth_header:
        try:
                mcp_client = MCPClient(lambda: streamablehttp_client(
                    url=gateway_url,
                    headers={"Authorization": auth_header}  
                ))
                
                with mcp_client:
                    tools = (
                        [
                            get_product_info,
                            get_return_policy,
                            get_technical_support
                        ]
                        + mcp_client.list_tools_sync()
                    )

                    memory_config = AgentCoreMemoryConfig(
                        memory_id=memory_id,
                        session_id=str(session_id),
                        actor_id=actor_id,
                        retrieval_config={
                            "support/customer/{actorId}/semantic/": RetrievalConfig(top_k=3, relevance_score=0.2),
                            "support/customer/{actorId}/preferences/": RetrievalConfig(top_k=3, relevance_score=0.2)
                        }
                    )

                    # Create the agent with all customer support tools
                    agent = Agent(
                        model=model,
                        tools=tools,
                        system_prompt=SYSTEM_PROMPT,
                        session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
                    )
                    # Invoke the agent
                    response = agent(user_input)
                    return response.message["content"][0]["text"]
        except Exception as e:
                print(f"MCP client error: {str(e)}")
                return f"Error: {str(e)}"
    else:
        return "Error: Missing gateway URL or authorization header"

if __name__ == "__main__":
    app.run()  #### AGENTCORE RUNTIME - LINE 4 ####
