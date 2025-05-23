"use client";

import { MessageSquareText, MessagesSquare, MoveUpRight } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";

const logItems = [
  {
    title: "Chat Completions",
    url: "/logs/chat-completions",
    icon: MessageSquareText,
  },
  {
    title: "Responses",
    url: "/logs/responses",
    icon: MessagesSquare,
  },
  {
    title: "Documentation",
    url: "https://llama-stack.readthedocs.io/en/latest/references/api_reference/index.html",
    icon: MoveUpRight,
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar>
      <SidebarHeader>
        <Link href="/">Llama Stack</Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Logs</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {logItems.map((item) => {
                const isActive = pathname.startsWith(item.url);
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      className={cn(
                        "justify-start",
                        isActive &&
                          "bg-gray-200 hover:bg-gray-200 text-primary hover:text-primary",
                      )}
                    >
                      <Link href={item.url}>
                        <item.icon
                          className={cn(
                            isActive && "text-primary",
                            "mr-2 h-4 w-4",
                          )}
                        />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
