"""Resolve network settings that must remain local to each Ray node."""

import os
import socket
from collections.abc import Mapping, MutableMapping
from typing import Any

import psutil


def ray_node_ipv4() -> str | None:
    """Return the IP that the local raylet registered for this node."""
    try:
        from ray.util import get_node_ip_address

        value = get_node_ip_address()
        if value and not value.startswith("127."):
            return value
    except Exception:
        pass
    return None


def interface_for_ipv4(
    ipv4: str | None,
    interface_addresses: Mapping[str, list[Any]] | None = None,
) -> str | None:
    """Find the interface that owns an exact IPv4 address."""
    if not ipv4:
        return None
    addresses_by_interface = psutil.net_if_addrs() if interface_addresses is None else interface_addresses
    for interface, addresses in addresses_by_interface.items():
        for address in addresses:
            if address.family == socket.AF_INET and address.address == ipv4:
                return interface
    return None


def configure_node_local_network(
    *,
    environ: MutableMapping[str, str] | None = None,
    node_ip: str | None = None,
    interface_addresses: Mapping[str, list[Any]] | None = None,
) -> dict[str, str | None]:
    """Set NCCL/Gloo identity from this actor's Ray node, never the driver."""
    target_environ = os.environ if environ is None else environ
    resolved_ip = node_ip or ray_node_ipv4()
    addresses_by_interface = psutil.net_if_addrs() if interface_addresses is None else interface_addresses
    interface = interface_for_ipv4(resolved_ip, addresses_by_interface)
    if not resolved_ip or not interface:
        known = {
            name: [addr.address for addr in addresses if addr.family == socket.AF_INET]
            for name, addresses in addresses_by_interface.items()
        }
        raise RuntimeError(
            "Could not map Ray's local node IP to a network interface before "
            f"NCCL initialization: ray_node_ip={resolved_ip!r}, interfaces={known!r}"
        )

    inherited_nccl_interface = target_environ.get("NCCL_SOCKET_IFNAME")
    inherited_gloo_interface = target_environ.get("GLOO_SOCKET_IFNAME")
    inherited_hostid = target_environ.get("NCCL_HOSTID")
    target_environ["NCCL_SOCKET_IFNAME"] = interface
    target_environ["GLOO_SOCKET_IFNAME"] = interface
    target_environ["NCCL_HOSTID"] = resolved_ip
    return {
        "interface": interface,
        "node_ip": resolved_ip,
        "inherited_nccl_interface": inherited_nccl_interface,
        "inherited_gloo_interface": inherited_gloo_interface,
        "inherited_hostid": inherited_hostid,
    }
