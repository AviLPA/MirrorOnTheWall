"""
Compatibility module for different versions of Pydantic.
This handles differences between Pydantic v1 and v2.
"""

import pkg_resources

# Get Pydantic version
try:
    pydantic_version = pkg_resources.get_distribution("pydantic").version
    major_version = int(pydantic_version.split('.')[0])
except:
    major_version = 1  # Default to v1 if can't determine version

if major_version >= 2:
    # Pydantic v2
    try:
        from pydantic.v1 import BaseModel, validator as v1_validator
        from pydantic import field_validator

        class PydanticCompatBase(BaseModel):
            """Base class compatible with both Pydantic v1 and v2"""
            model_config = {
                'arbitrary_types_allowed': True
            }

        def compat_validator(*args, **kwargs):
            """Compatibility wrapper for validators that works with both v1 and v2"""
            return field_validator(*args, **kwargs)

    except ImportError:
        print("WARNING: Error importing Pydantic v2 modules. Falling back to v1.")
        major_version = 1

if major_version < 2:
    try:
        # Fall back to Pydantic v1
        from pydantic import BaseModel, validator

        class PydanticCompatBase(BaseModel):
            """Base class compatible with both Pydantic v1 and v2"""
            class Config:
                """Configuration for Pydantic v1"""
                arbitrary_types_allowed = True

        def compat_validator(*args, **kwargs):
            """Compatibility wrapper for validators that works with both v1 and v2"""
            return validator(*args, **kwargs)

    except ImportError:
        # Create minimal fallback if pydantic is not available
        print("WARNING: Pydantic is not installed. Using fallback implementation.")
        
        class PydanticCompatBase:
            """Fallback base class when Pydantic is not available"""
            def __init__(self, **data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        def compat_validator(*args, **kwargs):
            """Fallback validator when Pydantic is not available"""
            def decorator(func):
                return func
            return decorator 