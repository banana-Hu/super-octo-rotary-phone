import { liveApi } from './live'
import { mockApi } from './mock'

export const isMockMode = import.meta.env.VITE_USE_MOCK !== 'false'
export const api = isMockMode ? mockApi : liveApi
export * from './types'
