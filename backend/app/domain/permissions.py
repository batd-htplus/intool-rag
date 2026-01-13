from enum import Enum

class Permission(str, Enum):
    VIEW_DOCUMENTS = "view_documents"
    UPLOAD_DOCUMENTS = "upload_documents"
    DELETE_DOCUMENTS = "delete_documents"
    MANAGE_PROJECTS = "manage_projects"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    "admin": [
        Permission.VIEW_DOCUMENTS,
        Permission.UPLOAD_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.MANAGE_PROJECTS
    ],
    "user": [
        Permission.VIEW_DOCUMENTS,
        Permission.UPLOAD_DOCUMENTS
    ],
    "viewer": [
        Permission.VIEW_DOCUMENTS
    ]
}

def has_permission(role: str, permission: Permission) -> bool:
    """Check if role has specific permission"""
    return permission in ROLE_PERMISSIONS.get(role, [])
