export const DISPOSAL_RULES = {
  plastic: {
    title: 'Plastic (Dry Waste)',
    description:
      'Rinse plastic containers and place them in the blue dry waste bin. Keep plastic separate from wet waste to improve recycling quality.'
  },
  paper: {
    title: 'Paper/Cardboard',
    description:
      'Keep paper and cardboard dry before placing in the blue recycling bin. Avoid mixing paper with oily or food-contaminated waste.'
  },
  organic: {
    title: 'Organic / Wet Waste',
    description:
      'Put food scraps and biodegradable waste in the green wet waste bin or compost pit. Remove plastic wrappers before disposal.'
  },
  biodegradable: {
    title: 'Organic / Wet Waste',
    description:
      'Place biodegradable waste in the green bin or compost system. Keep it free from plastic and metal contamination.'
  },
  metal: {
    title: 'Metal (Dry Recyclable)',
    description:
      'Rinse cans or metal containers and place them in the blue dry waste bin. Sharp metal items should be wrapped safely before disposal.'
  },
  glass: {
    title: 'Glass (Dry Recyclable)',
    description:
      'Place clean glass bottles and jars in the dry recycling stream. Wrap broken glass securely before handing over to collection workers.'
  },
  'e-waste': {
    title: 'E-Waste',
    description:
      'Do not throw e-waste into regular bins. Submit electronics to authorized e-waste collection centers or municipal drives.'
  },
  hazardous: {
    title: 'Hazardous / Medical Waste',
    description:
      'Do not mix hazardous waste with household bins. Follow local municipality and health department guidelines for safe disposal.'
  }
};

const DEFAULT_DISPOSAL_INFO = {
  title: 'General Waste Guidance',
  description:
    'Handle with care and check your local municipality guidelines for correct disposal.'
};

export const getDisposalInfo = (label) => {
  if (!label) {
    return DEFAULT_DISPOSAL_INFO;
  }

  const normalizedLabel = String(label).toLowerCase();
  return DISPOSAL_RULES[normalizedLabel] || DEFAULT_DISPOSAL_INFO;
};
