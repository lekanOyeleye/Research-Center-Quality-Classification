from pydantic import BaseModel


class ResearchCenterData(BaseModel):
    internalFacilitiesCount: float
    hospitals_10km: float
    pharmacies_10km: float
    facilityDiversity_10km: float
    facilityDensity_10km: float