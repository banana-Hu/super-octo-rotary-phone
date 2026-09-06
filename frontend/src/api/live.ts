import { apiClient } from './client'
import type { ComposeRequest, ComposeResponse, PublishRequest, PublishResponse, StartTaskRequest, StartTaskResponse, TaskStatus, UploadResponse, WorkListResponse } from './types'

export const liveApi = {
  async upload(files: File[]): Promise<UploadResponse> {
    const body = new FormData()
    files.forEach((file) => body.append('files', file))
    return (await apiClient.post<UploadResponse>('/upload', body)).data
  },
  async startTask(body: StartTaskRequest): Promise<StartTaskResponse> {
    return (await apiClient.post<StartTaskResponse>('/task/start', body)).data
  },
  async getTask(taskId: string): Promise<TaskStatus> {
    return (await apiClient.get<TaskStatus>(`/task/${encodeURIComponent(taskId)}/status`)).data
  },
  async compose(taskId: string, body: ComposeRequest): Promise<ComposeResponse> {
    return (await apiClient.post<ComposeResponse>(`/task/${encodeURIComponent(taskId)}/compose`, body)).data
  },
  async getWorks(): Promise<WorkListResponse> {
    return (await apiClient.get<WorkListResponse>('/works')).data
  },
  async publish(body: PublishRequest): Promise<PublishResponse> {
    return (await apiClient.post<PublishResponse>('/works', body)).data
  },
}
