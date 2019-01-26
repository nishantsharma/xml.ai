"""
This project is used to watch usage of GPU. If the GPU is not in use, the Ubuntu instance where
it is running is automatically shut down.

The intention is to save money  on unused cloud VMs.
"""
import sys, os, os.path, shutil, logging, time
from datetime import datetime, timedelta
from attrdict import AttrDict

from msrestazure.azure_active_directory import MSIAuthentication
from azure.mgmt.resource import ResourceManagementClient, SubscriptionClient
from azure.mgmt.compute import ComputeManagementClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#
#machineUsageWatchdogConfig = AttrDict({})
#machineUsageWatchdogConfig.vm_name = "gpu-vm"
#machineUsageWatchdogConfig.resource_group_name = "learn"

def azureInit():
    # Create MSI Authentication
    credentials = MSIAuthentication()

    # Create a Subscription Client
    subscription_client = SubscriptionClient(credentials)
    subscription = next(subscription_client.subscriptions.list())
    subscription_id = subscription.subscription_id

    # Create a Resource Management client
    resource_client = ResourceManagementClient(credentials, subscription_id)

    # Create a Compute client
    compute_client = ComputeManagementClient(credentials, subscription_id)

    return (subscription_client, compute_client)

def stopVirtualMachine(compute_client):
    async_vm_stop = compute_client.virtual_machines.deallocate(
        machineUsageWatchdogConfig.resource_group_name,
        machineUsageWatchdogConfig.vm_name)
    async_vm_stop.wait()

def touch(fname, times=None):
    fhandle = open(fname, 'a')
    try:
        os.utime(fname, times)
    finally:
        fhandle.close()
    os.chmod(fname, 666)

# Configuration
watchInterval = timedelta(minutes=2)
idleThresholdToSuspendInterval = timedelta(minutes=60)
gracePeriodAfterStartInterval = timedelta(minutes=20)
shutdownWatchdogFile="/var/run/MachineUsageWatchdog"
logFilename = "/var/log/MachineUsageWatchdog.log"

# Initialize logging.
logger = logging.getLogger('MachineUsageWatchdog')
hdlr = logging.FileHandler('/var/log/MachineUsageWatchdog.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.DEBUG)

# Initialize.
(subscription_client, compute_client) = azureInit()
serviceStartTime=datetime.now()
touch(shutdownWatchdogFile)
lofFile = open(logFilename, "a")
startTime = datetime.now()

# Application starts.
logger.info("Machine Usage watchdog service starts.\n")

# Main loop.
doesnt_exist_count = 0
while True:
    exists = os.path.exists(shutdownWatchdogFile)

    doesnt_exist_count = 0 if exists else doesnt_exist_count+1 

    if exists:
        lastModTime = datetime.fromtimestamp(os.path.getmtime(shutdownWatchdogFile))
        curTime = datetime.now()
        idlePeriodViolation = (curTime - lastModTime) > idleThresholdToSuspendInterval
        gracePeriodViolation = (curTime - startTime) > gracePeriodAfterStartInterval
        logger.info("Idle Period Gap {0}. Grace Period Gap {1}.\n".format(
                      (curTime-lastModTime - idleThresholdToSuspendInterval).total_seconds(),
                      (curTime-startTime - gracePeriodAfterStartInterval).total_seconds()))

    if doesnt_exist_count>=5 or (idlePeriodViolation and gracePeriodViolation):
        logger.warning("Suspending VM.\n")
        stopVirtualMachine(compute_client)
        logger.warning("VM waking up from suspension.\n")
        startTime = datetime.now()

    # Sleep for watch interval.
    time.sleep(watchInterval.seconds)
