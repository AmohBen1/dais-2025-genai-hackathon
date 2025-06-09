# %pip install -U llama-index llama-index-llms-databricks mlflow
# %pip install -U -qqq langchain_core langchain_databricks langchain_community
# %pip install -U typing_extensions
# %restart_python

import os
import time
import pandas as pd
import logging
from functools import wraps
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from databricks.sdk import WorkspaceClient
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from pyspark.sql import SparkSession
import requests
from typing import Dict, Any, List
import json

# Configure your personal access token

w = WorkspaceClient()

os.environ["DATABRICKS_HOST"] = w.config.host
os.environ["DATABRICKS_TOKEN"] = w.tokens.create(
    comment="for model serving", 
    lifetime_seconds=12000
).token_value

llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

class AccessibilityAgent:
    def __init__(self):
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(
            memory_key = "chat_history",
            return_messages = True,
            k = 5
        )
        self.spark = None
        self.max_retries = 2
        self.retry_delay = 1  # seconds
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _is_session_error(self, error):
        """Detect if error is related to session/connection issues"""
        error_str = str(error).lower()
        session_keywords = [
            'session_id is no longer usable',
            'inactivity_timeout', 
            'failed_precondition',
            'grpc error',
            'inactive_rpc_error',
            'connection closed',
            'bad_request: session_id'
        ]
        return any(keyword in error_str for keyword in session_keywords)

    def _execute_with_smart_retry(self, operation_func, operation_name="operation", *args, **kwargs):
        """Execute any operation with smart retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation_func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                
                if self._is_session_error(e):
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ Session error detected in {operation_name} (attempt {attempt + 1}). Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        return self._format_session_error_message(str(e), operation_name)
                else:
                    # Not a session error, don't retry
                    raise e
        
        # All retries failed
        return self._format_session_error_message(str(last_error), operation_name)
        
    def _format_session_error_message(self, error_details, operation_name="operation"):
        """Provide helpful error message when session fails"""
        return f"""**Spark Session Connection Lost{operation_name}**"""
    
    def _create_tools(self) -> List[Tool]:
        """Create a comprehensive tool for the AI agent"""

        tools = [
            Tool(
                name="sql_query_generator",
                description="Generates and executes SQL queries for the Airbnb dataset based on natural language requests. Use this when users ask for specific listings, filters, or data queries.",
                func = self._generate_and_execute_sql
            ),
            Tool(
                name="accessibility_analyzer",
                description="Analyzes property descriptions and reviews for accessibility features. Use this to evaluate accessibility of specific properties.",
                func = self._analyze_accesibility
            ),
            Tool(
                name="property_description",
                description="Returns the property description",
                func = self._get_property_description
            ),
            Tool(
                name="property_reviews",
                description="Returns the property reviews",
                func = self._get_property_reviews
            ),
            Tool(
                name = "hotel_price_fetcher",
                description= "Fetches hotel prices for comparison with Airbnb listings. Use when users want to compare accommodation options.",
                func = self._fetch_hotel_prices
            ),
            Tool(
                name = "neighhorhood_accessibility",
                description= "Gets neighborhood accessibility information. Use when users ask about area accessibility, public transport, or neighborhood features.",
                func = self._get_neighborhood_accessibility
            ),
            Tool(
                name = "advanced_search",
                description= "Performs advanced search combining multiple criteria and filters for accessible properties.",
                func = self._advanced_search
            )
        ]
        return tools
    
    def _test_spark_connection(self):
        """Test if Spark is working with a simple query"""
        try:
            result = spark.sql("SELECT 1 as connection_test").collect()
            return True, "Connection OK"
        except Exception as e:
            return False, str(e)

    def _execute_sql_operation(self, sql_query):
        """Execute SQL operation - this is what gets retried"""
        # First test connection
        is_connected, status = self._test_spark_connection()
        if not is_connected:
            raise Exception(f"Spark connection failed: {status}")
        
        # Execute the actual query
        return spark.sql(sql_query).to_pandas_on_spark()
    
    def _generate_and_execute_sql(self, user_request: str) -> str:
        """Generate SQL queries based on natural language and execute them"""

        sql_generation_prompt = PromptTemplate.from_template("""
            You are an SQL expert. Generate a SQL query for the Airbnb dataset based on the user request. 

            Available columns:
            - listing name, location, location_details, details, description, description_by_sections, reviews
            - host_name_of_reviews, guest_rating, price, amenities, property_type, beds, bedrooms, bathrooms, property_number_of_reviews, host, highlights, discount,is_supperhost, pricing_detials, reviews, travel_details,house_rules

            Table: sota_ai_agents.bright_initiative.airbnb_properties_information_csv
        
            User request: {request}
        
            Generate only the SQL query without explanation:
            """)
        

        try: 
            chain = sql_generation_prompt | self.llm | StrOutputParser()
            sql_query = chain.invoke({"request": user_request})
            
            # Clean up the SQL query (remove any markdown formatting)
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()

            # Execute the generated SQL
            result_df = self._execute_with_smart_retry(
            self._execute_sql_operation, 
            "SQL execution",
            sql_query
        )
            
            # Check if we got an error message instead of DataFrame
            if isinstance(result_df, str):
                return result_df 
        
            if result_df.empty:
                return "No results found for your query."
            
            return f"Query executed successfully. Found {len(result_df)} results:\n\n" + self._format_context(result_df)
            
        except Exception as e:
            if "session_id is no longer usable" in str(e):
                return ""
            
    def _get_property_description(self, property_id: str) -> str:
        """Get the description of a property based on its ID"""

        query = f"""
        SELECT 
            listing_name,
            description
        FROM sota_ai_agents.bright_initiative.airbnb_properties_information_csv
        WHERE property_id = '{property_id}'
        LIMIT 1
        """
        
        try:
            result_df = spark.sql(query).to_pandas_on_spark()
            return self._format_context(result_df)
        except Exception as e:
            return f"Property description not available: {str(e)}"
        
    def _get_property_reviews(self, property_id: str) -> str:
            """Get the reviews of a property based on its ID"""

            query = f"""
            SELECT 
                review_text,
                guest_rating,
                review_date
            FROM sota_ai_agents.bright_initiative.airbnb_properties_reviews
            WHERE property_id = '{property_id}'
            ORDER BY review_date DESC
            LIMIT 10
            """
            
            try:
                result_df = spark.sql(query).to_pandas_on_spark()
                return self._format_context(result_df)
            except Exception as e:
                return f"Property reviews not available: {str(e)}"
        
    def _analyze_accesibility(self, property_data: str) -> str:
        """Analyze property descriptions and reviews for accessibility features"""

        analysis_prompt = PromptTemplate.from_template("""
        You are an accessibility expert. Analyze the following property data and provie:
            1. Comfirmed accessibility features 
            2. Potential accessibility barriers 
            3. Accessibility score (0-10)
            4. Recommendatioins for the travelers with mobility needs
            
            Property data: {data}
            
            Provide a structured analysis: 
                
            """)

        chain = analysis_prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({"data": property_data})
        return analysis

    def _fetch_hotel_prices(self, location: str) -> str:
        """Fetch hotel prices from the bright_initiative dataset for comparison"""

        # SQL query to fetch hotel prices based on location
        sql_query = f"""
        SELECT title, price
        FROM sota_ai_agents.bright_initiative.booking_hotel_listings
        WHERE location = '{location}'
        """

        try:
            # Execute the SQL query
            results_df = spark.sql(sql_query).to_pandas_on_spark()
            # Convert the results to a list of dictionaries
            results_list = results_df.to_dict(orient='records')
            # Return the results as a JSON string
            return json.dumps(results_list, indent=2)
        
        except Exception as e:
            return f"Error executing SQL: {str(e)}\nGenerated SQL: {sql_query}"

    def _get_neighborhood_accessibility(self, location: str) -> str:
        """Get the accessibility rating of a neighborhood"""

        query = f"""
        SELECT 
            neighborhood
            public_transport_accessibility,
            sidewalk_quality,
            accessible_businesses_count,
            accessibility_rating
        FROM sota_ai_agents.bright_initiative.neighborhood_accessibility
        WHERE location ILIKE '%{location}%'
        LIMIT 5
        """
        
        try:
            result_df = spark.sql(query).to_pandas_on_spark()
            return self._format_context(result_df)
        except Exception as e:
            return f"Neighborhood data not available: {str(e)}"

    def _get_neighborhood_accessibility(self, location: str) -> str:
        """Get the accessibility rating of a neighborhood"""

        query = f"""
        SELECT 
            neighborhood,
            public_transport_accessibility,
            sidewalk_quality,
            accessible_businesses_count,
            accessibility_rating
        FROM sota_ai_agents.bright_initiative.neighborhood_accessibility
        WHERE location ILIKE '%{location}%'
        LIMIT 5
        """
        
        try:
            result_df = spark.sql(query).to_pandas_on_spark()
            return self._format_context(result_df)
        except Exception as e:
            return f"Neighborhood data not available: {str(e)}"
        
    def _advanced_search_operation(self, sql_query):
        """Advanced search operation - this is what gets retried"""
        return spark.sql(sql_query).to_pandas_on_spark()

    def _advanced_search(self, criteria: str) -> str:
        """Perform advanced search with multiple criteria"""
        
        # Parse criteria and build complex query
        search_prompt = PromptTemplate.from_template("""
        Generate an advanced SQL query that combines multiple criteria:
        
        Criteria: {criteria}
        
        Use these tables:
        - airbnb_properties_information_csv
        - neighborhood_accessibility  
        - reviews_sentiment_analysis
        
        Include JOINs where appropriate and complex WHERE conditions:
        """)

        try:
            chain = search_prompt | self.llm | StrOutputParser()
            sql_query = chain.invoke({"criteria": criteria})
            
            # Clean SQL
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            print(f"🔍 Generated advanced search SQL: {sql_query}")
            
            # Execute with auto-retry
            result_df = self._execute_with_smart_retry(
            self._advanced_search_operation,
            "Advanced search",
            sql_query
            )
            
            if isinstance(result_df, str):
                return result_df
            
            return self._format_context(result_df)
        
        except Exception as e:
            return f"Advanced search error: {str(e)}"

    def _format_context(self, df: pd.DataFrame) -> str:
        """Format DataFrame as JSON"""
        if df.empty:
            return "No results found."
        return df.to_json(orient='records', indent=2)

        
    def _create_agent(self):
        """Create the agent with tools and system message"""
        
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", """You are an advanced accessible travel assistant. You help users find accommodations 
        #     that meet their accessibility needs. 
            
        #     Your capabilities include:
        #     - Generating custom SQL queries for flexible searches
        #     - Analyzing accessibility features from descriptions and reviews
        #     - Comparing Airbnb listings with hotel alternatives
        #     - Providing neighborhood accessibility information
        #     - Performing complex searches with multiple criteria
            
        #     Always provide helpful, accurate information about accessibility features.
        #     When unsure about accessibility claims, clearly state limitations.
        #     Suggest multiple options when possible and explain the trade-offs."""),
        #     MessagesPlaceholder(variable_name="chat_history"),
        #     ("human", "{input}"),
        #     MessagesPlaceholder(variable_name="agent_scratchpad")
        # ])

        # Create a custom ReAct prompt
        react_prompt = PromptTemplate.from_template("""
            You are an advanced accessible travel assistant. Help users find accommodations that meet their accessibility needs.

            You have access to these tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought: {agent_scratchpad}
            """)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def chat(self, message: str) -> str:
        """Main chat interface"""
        try:
            response = self.agent.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"


# Usage Examples
def main():
    agent = AccessibilityAgent()
    
    # Example 1: Flexible SQL generation
    print("=== Example 1: Flexible Query ===")
    response1 = agent.chat("Find listings in Chicago with pools that have ratings above 4.5")
    print(response1)
    
    # Example 2: Complex accessibility analysis
    print("\n=== Example 2: Detailed Analysis ===")
    response2 = agent.chat("I need wheelchair accessible accommodations in San Francisco with good reviews mentioning elevators")
    print(response2)
    
    # # Example 3: Comparison with hotels
    # print("\n=== Example 3: Multi-option Comparison ===")
    # response3 = agent.chat("Compare accessible Airbnb options vs hotels in downtown Seattle for someone using a mobility scooter")
    # print(response3)
    
    # # Example 4: Neighborhood accessibility
    # print("\n=== Example 4: Neighborhood Analysis ===")
    # response4 = agent.chat("What's the accessibility situation in the Mission District of San Francisco?")
    # print(response4)

if __name__ == "__main__":
    main()