from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from timeManagement import SimulationClock
from eventManagement import EventQueue, EventType, Event
from stateManagement import StateManager
from orderManagement import OrderManager, OrderPriority
from costTracker import CostTracker
from inventoryPolices import BaseInventoryPolicy, PredictiveIntervalPolicy

@dataclass
class SimulationConfig:
    """Simulation configuration with all necessary parameters"""
    start_time: datetime
    actual_demand: List[float]
    inventory_policy: BaseInventoryPolicy
    forecast_demand: List[float] = None  # Single point predictions
    forecast_intervals: List[List[float]] = None  # Prediction intervals [prediction, lower, upper]
    initial_stock: float = None  # If None, will be calculated from policy
    
    def __post_init__(self):
        # Validate forecast inputs
        if self.forecast_demand is None and self.forecast_intervals is None:
            raise ValueError("Either forecast_demand or forecast_intervals must be provided")
        
        if self.forecast_intervals is not None:
            # Convert intervals to numpy array and validate shape
            self.forecast_intervals = np.array(self.forecast_intervals)
            if self.forecast_intervals.shape[1] != 3:
                raise ValueError("forecast_intervals must have shape (n, 3) [prediction, lower, upper]")
            
            # If single point forecast not provided, use predictions from intervals
            if self.forecast_demand is None:
                self.forecast_demand = self.forecast_intervals[:, 0].tolist()
        
        # Copy parameters from inventory policy
        self.lead_time = (self.inventory_policy.params.lead_time)*24
        self.review_period = self.inventory_policy.params.review_period
        self.service_level = self.inventory_policy.params.service_level


class IntegratedSimulator:
    def __init__(self, config: SimulationConfig):
        # Store configuration
        self.config = config
        
        # Initialize core components
        self.clock = SimulationClock(config.start_time)
        self.event_queue = EventQueue()
        self.order_manager = OrderManager(urgent_lead_time=24*2, non_urgent_lead_time=config.lead_time)
        self.cost_tracker = CostTracker()
        self.state_manager = StateManager(self.order_manager, self.cost_tracker)
        
        # Set up initial state
        self.state_manager.initialize_inventory(
            spare_part_id="SIMULATED_PART",
            location_id=1,
            initial_stock=config.initial_stock,
            reorder_point=0  # Will be calculated in first inventory check
        )
        
        # Schedule initial events
        self._schedule_initial_events()
    
    def _schedule_initial_events(self):
        """Schedule all demand events and initial inventory check"""
        # Schedule daily demand events using actual demand
        for day, demand in enumerate(self.config.actual_demand):
            event_time = self.config.start_time + timedelta(days=day)   
            self.event_queue.add_event(
                time=event_time,
                event_type=EventType.DEMAND,
                data={
                    'spare_part_id': "SIMULATED_PART",
                    'location_id': 1,
                    'quantity': demand,
                }
            )
        
        # Schedule initial inventory check
        self.event_queue.add_event(
            time=self.config.start_time,
            event_type=EventType.INVENTORY_CHECK
        )
    
    def _handle_demand_event(self, event: Event):
        """Process a demand event"""
        # Process demand and get unfulfilled quantity
        unfulfilled = self.state_manager.process_demand(
            spare_part_id=event.data['spare_part_id'],
            location_id=event.data['location_id'],
            quantity=event.data['quantity'],
            current_time=event.time
        )

        # Create urgent order for unfulfilled quantity plus expected demand until next non-urgent order arrives
        if unfulfilled > 0:
            # Fetch pending orders
            pending_orders = self.order_manager.get_pending_deliveries(event.time)

            # Find the earliest delivery due time of any pending non-urgent order
            earliest_non_urgent_due_time = None
            for order in pending_orders:
                if (order.priority == OrderPriority.NON_URGENT and 
                    order.spare_part_id == "SIMULATED_PART" and 
                    order.location_id == 1):
                    if earliest_non_urgent_due_time is None or order.delivery_due_time < earliest_non_urgent_due_time:
                        earliest_non_urgent_due_time = order.delivery_due_time

            # Calculate expected demand until the next non-urgent order arrives
            if earliest_non_urgent_due_time is not None:
                # Calculate days until arrival
                time_until_arrival = (earliest_non_urgent_due_time - event.time).total_seconds() / (24 * 3600)  # Convert to days
                days_until_arrival = int(np.ceil(max(time_until_arrival, 0)))  # Round up to next day, ensure non-negative

                # Use _get_forecast_window with exact days until arrival
                forecast_window = self._get_forecast_window(event.time, window_size=days_until_arrival)

                # Calculate expected demand based on forecast type
                if isinstance(self.config.inventory_policy, PredictiveIntervalPolicy):
                    expected_demand_until_arrival = np.sum(forecast_window[:, 0]) if len(forecast_window) > 0 else 0
                else:
                    expected_demand_until_arrival = np.sum(forecast_window) if len(forecast_window) > 0 else 0
            else:
                # Fallback if no non-urgent orders are pending: use non-urgent lead time
                L_non_urgent_days = int(self.order_manager.non_urgent_lead_time / 24)
                forecast_window = self._get_forecast_window(event.time, window_size=L_non_urgent_days)

                # Calculate expected demand based on forecast type
                if isinstance(self.config.inventory_policy, PredictiveIntervalPolicy):
                    expected_demand_until_arrival = np.sum(forecast_window[:, 0]) if len(forecast_window) > 0 else 0
                else:
                    expected_demand_until_arrival = np.sum(forecast_window) if len(forecast_window) > 0 else 0

            # Set urgent order quantity
            urgent_order_quantity = unfulfilled + expected_demand_until_arrival

            # Create urgent replenishment order
            order = self.order_manager.create_order(
                spare_part_id=event.data['spare_part_id'],
                location_id=event.data['location_id'],
                quantity=urgent_order_quantity,
                current_time=event.time,
                priority=OrderPriority.URGENT
            )

            # Calculate costs for urgent order
            self.cost_tracker.calculate_order_and_transport_costs(
                quantity=urgent_order_quantity,
                priority=OrderPriority.URGENT
            )

            # Schedule delivery event using urgent lead time
            delivery_time = event.time + timedelta(hours=self.order_manager.urgent_lead_time)
            self.event_queue.add_event(
                time=delivery_time,
                event_type=EventType.DELIVERY,
                data={'order': order}
            )

    def _handle_delivery_event(self, event: Event):
        """Process a delivery event"""
        order = event.data['order']
        self.state_manager.process_delivery(order, event.time)
        self.order_manager.deliver_order(order.order_id, event.time)
    
    def _get_forecast_window(self, current_time: datetime, window_size: int = 30) -> np.ndarray:
        """
        Get next window_size days of forecast data from current time
        Returns:
            - For PredictiveIntervalPolicy: Array of shape (n, 3) with [predictions, lower, upper]
            - For other policies: Array of shape (n,) with point predictions
        """
        days_elapsed = (current_time - self.config.start_time).days
        start_idx = days_elapsed
        end_idx = min(start_idx + window_size, len(self.config.forecast_demand))
        
        # Return appropriate forecast data based on policy type
        if isinstance(self.config.inventory_policy, PredictiveIntervalPolicy):
            if self.config.forecast_intervals is None:
                raise ValueError("PredictiveIntervalPolicy requires forecast intervals")
            return self.config.forecast_intervals[start_idx:end_idx]
        else:
            # For standard policy, return single point predictions
            return np.array(self.config.forecast_demand[start_idx:end_idx])
    
    def _handle_inventory_check(self, event: Event):
        """Process inventory check event"""
        current_state = self.state_manager.get_current_state("SIMULATED_PART", 1)

        if current_state.current_stock > 0:
            self.cost_tracker.calculate_holding_cost(current_state.current_stock)
        
        # Get forecast windows and recalculate inventory parameters
        forecast_window_30 = self._get_forecast_window(event.time, 30)
        forecast_window_365 = self._get_forecast_window(event.time, 365)
        
        if len(forecast_window_30) > 0:  # Only recalculate if we have forecast data
            # Update reorder point based on 30-day forecast
            new_reorder_point = self.config.inventory_policy.calculate_reorder_point(forecast_window_30)
            current_state.reorder_point = new_reorder_point
            
            # Check if we need to place an order using updated reorder point
            if current_state.current_stock <= current_state.reorder_point:
                
                # Check if there are any non-urgent orders already on the way
                pending_orders = self.order_manager.get_pending_deliveries(event.time)
                non_urgent_orders_pending = any(
                    order.priority == OrderPriority.NON_URGENT and 
                    order.spare_part_id == "SIMULATED_PART" and 
                    order.location_id == 1
                    for order in pending_orders
                )
                
                # Only place a new order if there are no non-urgent orders pending
                if not non_urgent_orders_pending:
                    order_quantity = self.config.inventory_policy.calculate_order_quantity(forecast_window_365)
                    
                    priority = OrderPriority.NON_URGENT
                    
                    # Create order with appropriate priority
                    order = self.order_manager.create_order(
                        spare_part_id="SIMULATED_PART",
                        location_id=1,
                        quantity=order_quantity,
                        current_time=event.time,
                        priority=priority
                    )
                    
                    # Get lead time based on priority from order manager
                    lead_time = (
                        self.order_manager.urgent_lead_time if priority == OrderPriority.URGENT 
                        else self.order_manager.non_urgent_lead_time
                    )
                    
                    # Schedule delivery event using priority-specific lead time
                    delivery_time = event.time + timedelta(hours=lead_time)
                    self.event_queue.add_event(
                        time=delivery_time,
                        event_type=EventType.DELIVERY,
                        data={'order': order}
                    )
                    # Calculate costs using same priority
                    self.cost_tracker.calculate_order_and_transport_costs(order_quantity, priority)
        
        # Schedule next inventory check based on policy's review period
        next_check = event.time + timedelta(days=self.config.review_period)
        if next_check < self.config.start_time + timedelta(days=len(self.config.actual_demand)):
            self.event_queue.add_event(
                time=next_check,
                event_type=EventType.INVENTORY_CHECK
            )
            #print(f"Scheduled next inventory check for {next_check}")
    
    def run(self) -> Dict:
        """Run the simulation and return results"""
        # print(f"Starting simulation at {self.clock.format_time()}")
        event_count = {
            EventType.DEMAND: 0,
            EventType.INVENTORY_CHECK: 0,
            EventType.DELIVERY: 0
        }
        
        while True:
            # Get next event
            event = self.event_queue.get_next_event()
            if not event:
                # print("\nFinal event counts:")
                for event_type, count in event_count.items():
                    # print(f"{event_type}: {count}")
                    pass
                break
                
            # Update simulation time
            self.clock.set_time(event.time)
            
            # print(event)
            # Debug logging
            #print(f"\nProcessing event: {event.event_type} at {event.time}")
            #print(f"Current stock level: {self.state_manager.get_current_state('SIMULATED_PART', 1).current_stock}")
            
            # Process event based on type
            if event.event_type == EventType.DEMAND:
                self._handle_demand_event(event)
                event_count[EventType.DEMAND] += 1
                #print(f"Processed demand event (Total: {event_count[EventType.DEMAND]})")
            elif event.event_type == EventType.INVENTORY_CHECK:
                self._handle_inventory_check(event)
                event_count[EventType.INVENTORY_CHECK] += 1
                #print(f"Processed inventory check event (Total: {event_count[EventType.INVENTORY_CHECK]})")
            elif event.event_type == EventType.DELIVERY:
                self._handle_delivery_event(event)
                event_count[EventType.DELIVERY] += 1
                #print(f"Processed delivery event (Total: {event_count[EventType.DELIVERY]})")
                #print(f"Warning: Unknown event type {event.event_type}")
            
            # Print queue size
            #print(f"Events remaining in queue: {self.event_queue.get_queue_size()}")
            
            # Update metrics
            self.state_manager.update_service_level(event.time)
            self.state_manager.update_costs(event.time)
        
        # print(f"Simulation completed at {self.clock.format_time()}")
        # print(event_count)
        
        # Compile and return results
        return self._get_results()
    
    def _get_results(self) -> Dict:
        """Compile simulation results including KPIs for costs, service level, and stockouts"""
        final_costs = self.cost_tracker.get_total_costs()
        final_state = self.state_manager.get_current_state("SIMULATED_PART", 1)
        
        # Calculate final service level
        latest_service_level = self.state_manager.service_level_history[-1][1] if self.state_manager.service_level_history else 0.0
        
        # Calculate total stockouts
        total_stockouts = final_state.lost_sales
        
        return {
            'kpis': {
                'total_costs': final_costs['total_cost'],
                'immediate_service_level': latest_service_level,  # Now based only on immediate fulfillment
                'total_stockouts': total_stockouts,
                'total_demand': final_state.total_demand,
                'immediate_fulfilled': final_state.immediate_fulfilled,
                'backorder_fulfilled': final_state.fulfilled_demand - final_state.immediate_fulfilled
            },
            'detailed_costs': final_costs,
            'service_level_history': self.state_manager.service_level_history,
            'final_stock': final_state.current_stock,
            'final_backorders': final_state.backorders,
            'stock_history': final_state.stock_history,
            'backorder_history': final_state.backorder_history,
            'stockout_history': final_state.lost_sales_history
        }
