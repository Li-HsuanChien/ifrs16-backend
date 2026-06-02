## TODO:
    API Call Feature: write a skeleton for api calls
    LLM Extraction calls API feature: Write a prompt 
    Extraction calls
    Calculations
    Evaluation calls from calculations
    Fix frontend data drill
    Fix frontend features:
        - local storage to view

##
    Frontend endpoints:
        Upload file

## Workflow:
    1. PDF/DOCX/DOC to text
    2. Extract and also evaluate details for following into Typescript specified structure:
    Clearer sections:
        - LeaseTermClauses
            - startDate
            - endDate
            - nonCancellablePeriod
            - renewalOption
            - reasonablyExpectedToExercise
            - earlyTermination
        - PaymentTerms
            - periodicRent
            - frequency
            - timing
            - prepayment
            - finalPayment
            - variableRent
            - variableRentPredictable
        - DiscountRate
            - implicitRate
            - incrementalBorrowingRate
        - ExemptionRules
            - shortTermLease
            - lowValueAsset
            - variableLeaseOnly
        - CostSeparation
            - leaseComponent
            - nonLeaseComponent
    Ambigious sections(Extract and also evaluate):
        - isSpecificAsset
        - hasControlRight
        - assetReplaceable
        - finalDecision
    3. calulations and balancing
        - 邏輯一致性檢查 
            - 檢查租期、付款金額、折現率等關鍵數據間的邏輯一致性
        - 金額異常檢測
            - 檢測異常的租金金額、折現率、總付款額等財務數據
        - 租期異常檢測
            - 檢查租期計算、續租條件、提前終止條款的合理性
        -    條款異常檢測
            - 檢測合約條款的完整性、一致性及與IFRS 16要求的符合度

